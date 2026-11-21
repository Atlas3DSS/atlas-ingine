"""Fused decode step: fixed buffers and a fixed op sequence for the 48 visits (contracts in SPEC.md).

Per token the residual stream is a set of block states plus the running partial block:

    B0 = embedding; B1 = hidden entering layer 6 (pass 0); B2[p] = hidden entering layer 12 in pass p;
    E[p] = hidden after layer 17 in pass p (E[2] is also the block entering layer 18).

Visit v = (layer l, pass p) runs: attention prologue (`mix`: apply the previous MLP write with its
ablation, attention-residual read(s), RMS norm; plus message gates in passes 1-2 of the loop) -> `proj` ->
`attn` -> `proj_conv` -> MLP prologue (`mix` with this visit's attention write) -> `router` -> `moe_up` ->
`moe_down`. The final read, norm and `head` produce logits.

`TorchOps` implements every op with PyTorch (slow; the wiring reference); `KernelOps` binds the CUDA
extensions. Both take the same arguments.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from . import model as M
from . import weights as W
from .weights import (BLOCK, CONV_K, D_REL, HEAD_DIM, HEADS, HIDDEN, KV_HEADS, LOOP_END, LOOP_START, NUM_VISITS,
                      TOP_K, VISITS, WINDOW)

SEG_ROWS = (("q", HEADS * HEAD_DIM), ("gate", HEADS * HEAD_DIM), ("rel", HEADS * D_REL), ("sink", HEADS),
            ("k", KV_HEADS * HEAD_DIM), ("v", KV_HEADS * HEAD_DIM), ("k_dyn", KV_HEADS * CONV_K),
            ("v_dyn", KV_HEADS * CONV_K))
SEG_WEIGHT = {"q": "wq", "gate": "wgate", "rel": "wr", "sink": "wsink", "k": "wk", "v": "wv",
              "k_dyn": "wk_dyn", "v_dyn": "wv_dyn"}
SEG_INPUT = {"q": 0, "gate": 0, "rel": 0, "sink": 0, "k": 1, "v": 2, "k_dyn": 1, "v_dyn": 2}


def proj_layout(full: bool, single_input: bool = False):
    """[(name, row_begin, row_end, input)] in SPEC order. `single_input`: q, k and v read the same vector
    (sliding layers outside loop passes 1-2)."""
    out, row = [], 0
    for name, n in SEG_ROWS:
        if name.endswith("_dyn") and not full:
            continue
        out.append((name, row, row + n, 0 if single_input else SEG_INPUT[name]))
        row += n
    return out


class TorchOps:
    """SPEC semantics in PyTorch, M = 1."""

    def __init__(self):
        self.lut = torch.tensor(M.E2M1, dtype=torch.float32, device="cuda")

    def mix(self, blocks, base, write, abl_r, abl_w, partial_out, queries, norm_w, out_bf16, out_f32):
        new = base if write is None else (write if base is None else base + write)
        if write is not None:
            new = new - abl_w * (write @ abl_r)[..., None] * abl_r
        if partial_out is not None:
            partial_out.copy_(new)
        if queries is None:  # layer 0: no attention residual
            mixed = new[None]
        else:
            mixed = M.block_attention(queries, list(blocks), new).transpose(0, 1)  # [Q, M, H]
        if out_f32 is not None:
            out_f32.copy_(mixed)
        if out_bf16 is not None:
            out_bf16.copy_(M.rms_norm(mixed, norm_w[:, None, :]).to(torch.bfloat16))

    def proj(self, xin, w, scale, layout, out):
        w = W.dequant(w, scale)
        for name, b, e, i in layout:
            out[:, b:e] = M.linear(xin[i].float(), w[b:e])

    def message_gates(self, layer, group, h, msg, norm_w, out_bf16, out_f32=None):
        """group "qkv" (attention inputs, G = 3) or "mlp1" (MLP input, G = 1)."""
        for g, name in enumerate(("q", "k", "v") if group == "qkv" else ("mlp",)):
            gated = M.message_gate(layer.t, name, h[g], msg[g])
            if out_f32 is not None:
                out_f32[g].copy_(gated)
            out_bf16[g].copy_(M.rms_norm(gated, norm_w[g]).to(torch.bfloat16))

    def attn(self, layer, v, st, proj, layout, pos, out):
        t = layer.t
        seg = {name: proj[:, b:e] for name, b, e, _ in layout}
        T = proj.shape[0]
        q = (seg["q"].view(T, HEADS, HEAD_DIM) * t["q_scale"])
        q = M.l2_clip(q, W.QK_CLIP).to(torch.bfloat16).float()
        rel = M.l2_clip(seg["rel"].view(T, HEADS, D_REL) * t["r_scale"], W.REL_CLIP).to(torch.bfloat16).float()
        if layer.full:
            tau = M.log_scale(pos)[:, None, None]
            q = (q * tau).to(torch.bfloat16).float()
            rel = (rel * tau).to(torch.bfloat16).float()
            k, new_k = M.dynamic_conv(seg["k"], st.conv_k[v], t["k_static"], seg["k_dyn"].view(T, KV_HEADS, CONV_K))
            vv, new_v = M.dynamic_conv(seg["v"], st.conv_v[v], t["v_static"], seg["v_dyn"].view(T, KV_HEADS, CONV_K))
        else:
            k, new_k = M.static_conv(seg["k"], st.conv_k[v], t["k_conv"], math.sqrt(CONV_K + 1))
            vv, new_v = M.static_conv(seg["v"], st.conv_v[v], t["v_conv"], math.sqrt(CONV_K + 1))
        st.conv_k[v].copy_(new_k)
        st.conv_v[v].copy_(new_v)
        k = M.l2_clip(k.to(torch.bfloat16).float().view(T, KV_HEADS, HEAD_DIM), W.QK_CLIP).to(torch.bfloat16)
        vv = M.rms_norm(vv.to(torch.bfloat16).float().view(T, KV_HEADS, HEAD_DIM)).to(torch.bfloat16)
        rel_logits = (rel.to(torch.bfloat16) @ t["rel_proj"].to(torch.bfloat16)).float()
        if layer.full:
            fi = st.full_index[v]
            st.write_full(fi, pos, k, vv)
            keys, values = st.k_full[fi], st.v_full[fi]
            kpos = torch.arange(st.max_ctx, device=proj.device)
            dist = pos[:, None] - kpos[None, :]
            valid = dist >= 0
        else:
            ri = st.ring_index[v]
            slot = pos % WINDOW
            st.k_ring[ri].index_copy_(0, slot, k)
            st.v_ring[ri].index_copy_(0, slot, vv)
            st.ring_pos[ri].index_copy_(0, slot, pos)
            keys, values, kpos = st.k_ring[ri], st.v_ring[ri], st.ring_pos[ri]
            dist = pos[:, None] - kpos[None, :]
            valid = (kpos[None, :] >= 0) & (dist >= 0) & (dist < WINDOW)
        o = M.Model._attend(q, keys, values, dist, valid, rel_logits, seg["sink"])
        gate = torch.sigmoid(seg["gate"])
        out.copy_((o.to(torch.bfloat16).float() * gate).to(torch.bfloat16))

    def proj_conv(self, x, w, scale, conv_w, conv_state, out, conv_hist=None):
        y = M.linear(x.float(), w, scale)
        if conv_hist is not None:
            conv_hist.copy_(y)
        y, new = M.static_conv(y, conv_state, conv_w, math.sqrt(CONV_K + 1))
        conv_state.copy_(new)
        out.copy_(y)

    def router(self, x, w, bias, idx, wts):
        scores = torch.sigmoid(x.float() @ w.float().T)
        top = torch.topk(scores + bias, TOP_K, dim=-1, sorted=False).indices
        s = scores.gather(1, top)
        idx.copy_(top.int())
        wts.copy_(s / (s.sum(-1, keepdim=True) + 1e-20) * W.ROUTE_SCALE)

    def moe_up(self, x, idx, experts, out):
        w13 = M.decode_experts(experts, "w13", idx[0].long(), self.lut)
        gu = torch.bmm(w13, x[0].expand(TOP_K, -1)[..., None]).squeeze(-1).to(torch.bfloat16).float()
        out[0].copy_(M.Model._act(gu[:, 0::2], gu[:, 1::2]).to(torch.bfloat16))

    def moe_down(self, act, idx, wts, experts, conv_w, conv_state, out, conv_hist=None):
        w2 = M.decode_experts(experts, "w2", idx[0].long(), self.lut)
        down = torch.bmm(w2, act[0][..., None]).squeeze(-1).to(torch.bfloat16).float()
        s = (down * wts[0, :, None]).sum(0, keepdim=True).to(torch.bfloat16).float()
        if conv_hist is not None:
            conv_hist.copy_(s)
        y, new = M.static_conv(s, conv_state, conv_w, math.sqrt(CONV_K + 1))
        conv_state.copy_(new)
        out.copy_(y)

    def head(self, x, w, scale, out):
        out.copy_(M.softcap(M.linear(x.float(), w, scale), W.LOGIT_SOFTCAP))

    def embed(self, token, table, norm_w, out):
        out.copy_(M.rms_norm(table[token].float(), norm_w).to(torch.bfloat16).float())


class KernelOps(TorchOps):
    """CUDA kernels where available (see engine/csrc/<area>); TorchOps for the rest.

    With `pdl`, kernels use programmatic dependent launch: a kernel may start (and prefetch weights)
    while its predecessor finishes, and waits for it before reading its outputs."""

    def __init__(self, pdl: bool = True):
        super().__init__()
        from . import ext
        self.pdl = pdl
        self.moe = ext.load("moe")
        self.dense = ext.load("dense")
        self.attn_ext = ext.load("attn")
        self.none_f32 = torch.empty(0, device="cuda")
        self.none_bf16 = torch.empty(0, dtype=torch.bfloat16, device="cuda")
        self.none_i64 = torch.empty(0, dtype=torch.long, device="cuda")
        # one workspace serves every stream-ordered attention call; zeroed once (the kernel resets its counter)
        self.attn_ws = torch.zeros(self.attn_ext.attn_workspace_bytes(8), dtype=torch.uint8, device="cuda")
        self._ptrs: dict[tuple, torch.Tensor] = {}
        self._segments: dict[tuple, torch.Tensor] = {}

    def attn(self, layer, v, st, proj, layout, pos, out):
        t = layer.t
        if layer.full:
            fi = st.full_index[v]
            self.attn_ext.attn_decode(proj, t["q_scale"], t["r_scale"], t["rel_proj"], t["k_static"], t["v_static"],
                                      st.conv_k[v], st.conv_v[v], st.k_full[fi], st.v_full[fi], self.none_i64, pos,
                                      0, True, self.attn_ws, out)
        else:
            ri = st.ring_index[v]
            self.attn_ext.attn_decode(proj, t["q_scale"], t["r_scale"], t["rel_proj"], t["k_conv"], t["v_conv"],
                                      st.conv_k[v], st.conv_v[v], st.k_ring[ri], st.v_ring[ri], st.ring_pos[ri], pos,
                                      WINDOW, False, self.attn_ws, out)

    def _opt(self, t, bf16=False):
        return t if t is not None else (self.none_bf16 if bf16 else self.none_f32)

    def _block_ptrs(self, blocks):
        """Device tensor of block-state pointers. Built on first use (the warm-up before graph capture), so
        capture only replays existing tensors."""
        key = tuple(b.data_ptr() for b in blocks)
        if key not in self._ptrs:
            self._ptrs[key] = torch.tensor(key, dtype=torch.long, device="cuda")
        return self._ptrs[key]

    def mix(self, blocks, base, write, abl_r, abl_w, partial_out, queries, norm_w, out_bf16, out_f32):
        self.dense.mix(self._block_ptrs(blocks), self._opt(base), self._opt(write),
                       abl_r if write is not None else self.none_f32, abl_w if write is not None else self.none_f32,
                       self._opt(partial_out), self._opt(queries), self._opt(norm_w), self._opt(out_bf16, True),
                       self._opt(out_f32), self.pdl)

    def proj(self, xin, w, scale, layout, out):
        key = tuple(layout)
        if key not in self._segments:
            self._segments[key] = torch.tensor([(b, e, i) for _, b, e, i in layout], dtype=torch.int32, device="cuda")
        self.dense.proj(xin, w, self._opt(scale), self._segments[key], out, self.pdl)

    def proj_conv(self, x, w, scale, conv_w, conv_state, out, conv_hist=None):
        self.dense.proj_conv(x, w, self._opt(scale), conv_w, conv_state, out, self._opt(conv_hist), self.pdl)

    def router(self, x, w, bias, idx, wts):
        self.dense.router(x, w, bias, idx, wts, self.pdl)

    def message_gates(self, layer, group, h, msg, norm_w, out_bf16, out_f32=None):
        t = layer.t
        self.dense.message_gates(h, msg, t[f"{group}_gate_down"], t[f"{group}_gate_forget16"],
                                 t[f"{group}_gate_gain16"], norm_w, out_bf16, self._opt(out_f32), self.pdl)

    def head(self, x, w, scale, out):
        self.dense.head(x, w, self._opt(scale), out, self.pdl)

    def embed(self, token, table, norm_w, out):
        self.dense.embed(token, table, norm_w, out, self.pdl)

    def moe_up(self, x, idx, experts, out):
        self.moe.moe_up(x, idx, experts.w13, experts.w13_scale, experts.w13_factor,
                        0 if experts.fmt == "nvfp4" else 1, out, self.pdl)

    def moe_down(self, act, idx, wts, experts, conv_w, conv_state, out, conv_hist=None):
        # with PDL, moe_down prefetches weights before waiting: valid because moe_up always precedes it
        self.moe.moe_down(act, idx, wts, experts.w2, experts.w2_scale, experts.w2_factor,
                          0 if experts.fmt == "nvfp4" else 1, conv_w, conv_state, out, self._opt(conv_hist),
                          self.pdl)


class Buffers:
    """Fixed device buffers for one call width M (one set per captured CUDA graph)."""

    def __init__(self, m: int):
        dev = "cuda"
        z = lambda *s, dt=torch.float32: torch.zeros(*s, dtype=dt, device=dev)  # noqa: E731
        H = HIDDEN
        self.m = m
        self.B0, self.B1 = z(m, H), z(m, H)
        self.B2 = [z(m, H) for _ in range(W.PASSES)]
        self.E = [z(m, H) for _ in range(W.PASSES)]
        self.hid, self.part, self.y, self.y2 = z(m, H), z(m, H), z(m, H), z(m, H)
        self.xin = z(3, m, H, dt=torch.bfloat16)
        self.mixf = z(3, m, H)
        self.msgs = z(4, m, H)
        self.xm = z(1, m, H, dt=torch.bfloat16)
        self.mixm = z(1, m, H)
        self.proj_out = {e: z(m, e) for e in {proj_layout(f)[-1][2] for f in (False, True)}}
        self.attn_out = z(m, HEADS * HEAD_DIM, dt=torch.bfloat16)
        self.idx = z(m, TOP_K, dt=torch.int32)
        self.wts = z(m, TOP_K)
        self.act = z(m, TOP_K, W.INTER, dt=torch.bfloat16)
        self.xf = z(1, m, H, dt=torch.bfloat16)
        self.logits = z(m, W.UNPADDED_VOCAB)
        # DSpark features of each row: partial block entering visits 5, 23, 41 and the final normed hidden
        self.taps = z(m, len(M.TAP_VISITS) + 1, H, dt=torch.bfloat16)
        # convolution inputs of each row, per visit, for rolling back rejected rows (verification only)
        kv = KV_HEADS * HEAD_DIM
        self.hist = {"attn": z(NUM_VISITS, m, H), "mlp": z(NUM_VISITS, m, H),
                     "k": z(NUM_VISITS, m, kv), "v": z(NUM_VISITS, m, kv)}

    def blocks(self, l: int, p: int):
        """Completed blocks visible to layer l's attention read in pass p (before its own boundary append)."""
        if l < 1:
            return []
        if l < 7:
            return [self.B0]
        if l < 13:
            return [self.B0, self.B1]
        if l < 19:
            return [self.B0, self.B1, self.B2[p if l < LOOP_END else 2]]
        return [self.B0, self.B1, self.B2[2], self.E[2]]


class FastDecoder:
    def __init__(self, weights: W.Weights, ablation: M.Ablation, state: M.State, ops=None, dense_fp8: bool = False):
        """`dense_fp8`: store the dense matrices of the decode step as FP8 rows on the shared `weights`, in
        place (the PyTorch prefill then reads the same values). Weights converted earlier stay FP8."""
        self.w, self.abl, self.st = weights, ablation, state
        self.ops = ops or TorchOps()
        self.bufs: dict[int, Buffers] = {}
        self._concat_projections()
        self._stack_layer_params()
        if dense_fp8:
            self._fp8_dense()
        self.final_norm1 = self.w.final_norm[None]
        # dense matrices of the decode step: (weight, per-row scale or None)
        self.proj_w = [(l.t["wproj"], l.scales.get("wproj")) for l in self.w.layers]
        self.out_w = [(l.t["wo"], l.scales.get("wo")) for l in self.w.layers]
        self.head_w = (self.w.unembed, self.w.unembed_scale)

    def _concat_projections(self):
        """One [R, H] matrix per layer in SPEC row order; the per-projection tensors become views into it
        so the prefill path shares the memory. Idempotent."""
        for layer in self.w.layers:
            if "wproj" in layer.t:
                continue
            layout = proj_layout(layer.full)
            cat = torch.cat([layer.t[SEG_WEIGHT[name]] for name, _, _, _ in layout])
            for name, b, e, _ in layout:
                layer.t[SEG_WEIGHT[name]] = cat[b:e]
            layer.t["wproj"] = cat

    def _fp8_dense(self):
        """`wproj` (with its per-projection views), `wo` and the unembedding -> FP8 rows + FP32 row scales
        (`fp8_rows`), replacing the BF16 tensors so only one copy stays resident. Idempotent."""
        from . import ext
        fp8 = ext.load("dense").fp8_rows
        for layer in self.w.layers:
            t, sc = layer.t, layer.scales
            if "wproj" in sc:
                continue
            for name in ("wproj", "wo"):
                t[name], sc[name] = fp8(t[name])
            for name, b, e, _ in proj_layout(layer.full):
                t[SEG_WEIGHT[name]], sc[SEG_WEIGHT[name]] = t["wproj"][b:e], sc["wproj"][b:e]
        if self.w.unembed_scale is None:
            self.w.unembed, self.w.unembed_scale = fp8(self.w.unembed)
        torch.cuda.empty_cache()  # return the freed BF16 blocks to the driver

    def _stack_layer_params(self):
        """Per-layer [G, ...] stacks the kernels take (norm weights per query, message-gate weights per gate);
        the per-gate tensors become views into the stacks."""
        for layer in self.w.layers:
            t = layer.t
            if layer.full:
                t["attn_norm3"] = torch.stack([t["q_norm"], t["k_norm"], t["v_norm"]])
                t["attn_norm_mix"] = t["attn_norm3"]
            else:
                t["attn_norm3"] = t["attn_norm"][None].expand(3, -1).contiguous()
                t["attn_norm_mix"] = t["attn_norm"][None]
            t["mlp_norm1"] = t["mlp_norm"][None]
            t["router_bf16"] = t["router"].to(torch.bfloat16)  # checkpoint values are BF16: exact
            if layer.loop:
                for part in ("down", "forget", "gain"):
                    qkv = torch.stack([t[f"{g}_gate_{part}"] for g in "qkv"])
                    for i, g in enumerate("qkv"):
                        t[f"{g}_gate_{part}"] = qkv[i]
                    t[f"qkv_gate_{part}"] = qkv
                    t[f"mlp1_gate_{part}"] = t[f"mlp_gate_{part}"][None]
                for group in ("qkv", "mlp1"):  # BF16 copies for the kernels (exact, half the bytes)
                    for part in ("forget", "gain"):
                        t[f"{group}_gate_{part}16"] = t[f"{group}_gate_{part}"].to(torch.bfloat16)

    def buffers(self, m: int) -> Buffers:
        if m not in self.bufs:
            self.bufs[m] = Buffers(m)
        return self.bufs[m]

    @torch.inference_mode()
    def decode(self, token: torch.Tensor, pos: torch.Tensor, history: bool = False) -> torch.Tensor:
        """Runs M = len(token) consecutive tokens starting at position pos[0]; returns logits [M, V] (a view
        of the call width's fixed buffer). The DSpark features land in `buffers(M).taps`; with `history`, the
        convolution inputs of every row land in `buffers(M).hist` for `rollback`."""
        ops, w, st, abl = self.ops, self.w, self.st, self.abl
        b = self.buffers(token.shape[0])
        ops.embed(token, w.embed, w.embed_norm, b.B0)
        prev_v = None
        tap = 0
        for v, (l, p) in enumerate(VISITS):
            layer = w.layers[l]
            t = layer.t
            loop = LOOP_START <= l < LOOP_END
            msgs = loop and p > 0
            boundary = l % BLOCK == 0
            blocks = b.blocks(l, p)
            # ---- hidden entering this visit (the previous MLP write applied), stored where it is a block
            if v == 0:
                hid, base, write = b.B0, b.B0, None
            elif l == LOOP_START and p > 0:
                hid, base, write = b.B1, b.B1, None
            else:
                hid = {6: b.B1, 12: b.B2[p], 18: b.E[2]}.get(l, b.hid)
                base, write = b.part, b.y2
            r_prev = abl.mlp_dirs[prev_v] if prev_v is not None else abl.mlp_dirs[0]
            w_prev = abl.mlp_w[prev_v:prev_v + 1] if prev_v is not None else abl.mlp_w[0:1]
            if msgs:
                i = l - LOOP_START
                src_base, src_write = (b.part, b.y2) if i == 0 else (b.E[p - 1], None)
                # messages from pass p-1: its layer-12 block and its final hidden (E[p-1], completed at i == 0)
                ops.mix([b.B2[p - 1]], src_base, src_write, r_prev, w_prev,
                        b.E[p - 1] if i == 0 else None, w.message_queries[i], None, None, b.msgs)
                if i == 0:
                    write = None  # pass p-1's last write went into E[p-1]; layer 6 restarts from B1
            # ---- attention read(s)
            queries = t.get("attn_query")
            nq = 3 if layer.full else 1
            if msgs:
                ops.mix(blocks, base, write, r_prev, w_prev, hid if hid is not base else None, queries, None, None,
                        b.mixf[:nq])
                if not layer.full:
                    b.mixf[1:].copy_(b.mixf[:1].expand(2, -1, -1))
                ops.message_gates(layer, "qkv", b.mixf, b.msgs[:3], t["attn_norm3"], b.xin)
            else:
                ops.mix(blocks, base, write, r_prev, w_prev, hid if hid is not base else None,
                        queries if l > 0 else None, t["attn_norm_mix"], b.xin[:nq], None)
            if v in M.TAP_VISITS:
                b.taps[:, tap].copy_(hid)
                tap += 1
            # ---- attention
            layout = proj_layout(layer.full, single_input=not layer.full and not msgs)
            proj_out = b.proj_out[layout[-1][2]]
            ops.proj(b.xin, *self.proj_w[l], layout, proj_out)
            if history:
                seg = {name: (s, e) for name, s, e, _ in layout}
                b.hist["k"][v].copy_(proj_out[:, seg["k"][0]:seg["k"][1]])
                b.hist["v"][v].copy_(proj_out[:, seg["v"][0]:seg["v"][1]])
            ops.attn(layer, v, st, proj_out, layout, pos, b.attn_out)
            ops.proj_conv(b.attn_out, *self.out_w[l], t["attn_conv"], st.conv_attn[v], b.y,
                          b.hist["attn"][v] if history else None)
            # ---- MLP read: this visit's attention write, ablated, on top of hid (or a fresh block)
            mlp_blocks = blocks + [hid] if boundary else blocks
            if msgs:
                ops.mix(mlp_blocks, None if boundary else hid, b.y, abl.attn_dirs[v], abl.attn_w[v:v + 1],
                        b.part, t["mlp_query"], None, None, b.mixm)
                ops.message_gates(layer, "mlp1", b.mixm, b.msgs[3:4], t["mlp_norm1"], b.xm)
            else:
                ops.mix(mlp_blocks, None if boundary else hid, b.y, abl.attn_dirs[v], abl.attn_w[v:v + 1],
                        b.part, t["mlp_query"], t["mlp_norm1"], b.xm, None)
            ops.router(b.xm[0], t["router_bf16"], t["router_bias"], b.idx, b.wts)
            ops.moe_up(b.xm[0], b.idx, layer.experts, b.act)
            ops.moe_down(b.act, b.idx, b.wts, layer.experts, t["mlp_conv"], st.conv_mlp[v], b.y2,
                         b.hist["mlp"][v] if history else None)
            prev_v = v
        # ---- final read
        ops.mix([b.B0, b.B1, b.B2[2], b.E[2]], b.part, b.y2, abl.mlp_dirs[prev_v],
                abl.mlp_w[prev_v:prev_v + 1], None, w.final_query, self.final_norm1, b.xf, None)
        b.taps[:, tap].copy_(b.xf[0])
        ops.head(b.xf[0], *self.head_w, b.logits)
        return b.logits

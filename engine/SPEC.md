# ingine decode kernels: contracts

ingine runs Inkling-2-Mini on one RTX 5090 (sm_120a, 32 GB, ~1.79 TB/s). Prefill is PyTorch; the decode
step (one token, later up to 8 consecutive tokens for speculative verification) is a fixed sequence of
custom kernels captured in one CUDA graph. This file is the contract between the kernels and the host code
in `ingine/fast.py` (owned by the integrator).

The mathematical oracle is `ingine/model.py` (PyTorch, FP32 activations, BF16 matrix inputs), itself
checked against the transformers reference. Weights come from `ingine/weights.py`;
`weights.load_layer(model_dir, l)` loads one layer (~0.8 GB GPU) for tests.

## Model facts used by kernels

* H = 2304, 24 query heads, 8 KV heads, head_dim 128 (GQA 3:1), d_rel 16, rel_extent 1024, window 1024.
* 48 layer visits per token: layers 0-5, layers 6-17 three times, layers 18-23. Full-attention layers are
  5, 11, 17, 23; all others use a 1024-token sliding window.
* MoE: 128 experts, top-8, expert intermediate I = 1152. Experts of layers 0-19 are NVFP4, 20-23 MXFP8
  (layouts in `weights.py` docstring). `w13` rows are interleaved: row 2i = gate i, row 2i+1 = up i.
* All short convolutions have kernel 4 and keep the previous 3 inputs as state ``[3, C]`` (oldest first,
  zeros before the first token).

## Conventions (all kernels)

* One pybind11 extension per area: `engine/csrc/<area>/` (`.cu` + `.cpp`), built by `ingine/ext.py`
  (`ext.load("<area>")`). Shared helpers may go in `engine/csrc/common/<area>_*.cuh`; do not edit another
  area's files.
* Every op takes preallocated outputs and workspaces, launches on `at::cuda::getCurrentCUDAStream()`, never
  allocates, never synchronises, and reads positions/lengths from device memory, so it can be captured in
  a CUDA graph and replayed with different contents.
* `M` = tokens per call (1 ≤ M ≤ 8), consecutive positions. Correctness is required for every M in that
  range; performance matters first for M = 1.
* Matrix inputs are BF16; accumulation and activations are FP32. Where the reference rounds an
  intermediate to BF16 (noted below), kernels round it too; everything else may stay FP32.
* Qualification: an independent FP32/FP64 oracle built from `model.py` functions with decoded weights, at
  real model shapes and real weights from several layers (cover NVFP4 and MXFP8; sliding and full;
  loop and non-loop as relevant). Report max abs / relative error. Tests: `engine/tests/test_<area>.py`
  (pytest, GPU memory < 3 GB). Benchmarks: `engine/bench/bench_<area>.py` (CUDA events, warm-up, rotate
  over several layers' weights so L2 does not serve them; report µs and achieved GB/s of weight bytes).
* The GPU is shared with a long-running ~22 GB job until the integrator says otherwise: keep memory low and
  treat timings as noisy (repeat, take medians, state the condition).

## Area `moe` (routed experts)

`moe_up(x, idx, w13, w13_scale, w13_factor, fmt, out)`

* `x` bf16 `[M, H]` (normalised MLP input), `idx` int32 `[M, 8]`, `out` bf16 `[M, 8, I]`.
* `fmt` 0 = NVFP4 (`w13` uint8 `[128, 2I, H/2]`, `w13_scale` float8_e4m3 `[128, 2I, H/16]`, `w13_factor`
  fp32 `[128]`), 1 = MXFP8 (`w13` float8_e4m3 `[128, 2I, H]`, `w13_scale` uint8 E8M0 `[128, 2I, H/32]`,
  factor unused).
* Per (m, e = idx[m, k]): `gu = W13_e · x[m]` (decoded weight value = code × block scale × factor, as
  `model.decode_experts`); gate = gu[2i], up = gu[2i+1] each rounded to BF16 (reference BF16 linear
  output); `out[m, k, i] = bf16(softcap(gate, 4) * sigmoid(gate) * softcap(up, 25))`,
  `softcap(x, c) = c * tanh(x / c)`.

`moe_down(act, idx, wts, w2, w2_scale, w2_factor, fmt, conv_w, conv_state, out)`

* `act` bf16 `[M, 8, I]`, `wts` fp32 `[M, 8]`, `w2` NVFP4 uint8 `[128, H, I/2]` + e4m3 `[128, H, I/16]` +
  factor, or MXFP8 e4m3 `[128, H, I]` + E8M0 `[128, H, I/32]`.
* `s[m] = bf16( Σ_k wts[m, k] * bf16(W2_e · act[m, k]) )` (per-expert output rounded to BF16, FP32 sum,
  sum rounded to BF16).
* Post-MLP convolution per channel c (`conv_w` fp32 `[H, 4]`, conv1d order: tap 3 multiplies the current
  token): `ext = [conv_state[0..2], s[0..M-1]]`, `out[m, c] = (Σ_j conv_w[c, j] * ext[m + j] + s[m]) / √5`.
  Write `conv_state = ext[M..M+2]` (the last three inputs). `out` fp32 `[M, H]`.
* Target: ≥ 80% of DRAM bandwidth on the 8 experts' bytes per token (NVFP4 ≈ 4.5 MB per expert, MXFP8 ≈
  8.2 MB) for M = 1.

## Area `attn` (attention core)

`attn_decode(proj, layer params, conv_k_state, conv_v_state, k_cache, v_cache, kpos, pos, window,
log_scaling, workspace, out)`

* `proj` fp32 `[M, R]`, segments in this row order: q 3072 | gate 3072 | rel 384 | sink 24 | k 1024 |
  v 1024 | (full layers only) k_dyn 32 | v_dyn 32. Each value is a BF16-rounded projection output.
* Per token (positions `p = pos[0] + m`):
  * `q = clip(q.view(24,128) * q_scale, 22.627)`, rounded to BF16; full layers then `q *= tau(p)` and round
    again. `clip(x, t) = x * min(1, t / (‖x‖ + 1e-6))` per head; `tau(p) = 1 + 0.1 * ln(max((p+1)/8192, 1))`.
  * `rel = clip(rel.view(24,16) * r_scale, 8.0)`, BF16; full layers `rel *= tau(p)`, BF16;
    `rel_logits[h, d] = bf16(Σ_j rel[h, j] * bf16(rel_proj[j, d]))`, d < 1024.
  * Keys/values: sliding layers use the static conv (`k_conv`/`v_conv` fp32 `[1024, 4]`, conv1d order,
    `(conv + x)/√5`); full layers use the dynamic conv: tap weight for lag ℓ (0 = current) of channel c in
    head h is `k_static[ℓ, c] + k_dyn[h*4 + ℓ]`, output `(Σ_ℓ w_ℓ x_{t-ℓ} + x_t)/3`. Conv states as above
    (`[3, 1024]`, inputs are the BF16-rounded projections). Then round to BF16; keys:
    `clip(k.view(8,128), 22.627)`; values: per-head RMS norm without weight (eps 1e-6). Store into the cache
    (dtype below).
  * Cache: full layers write position p at `k_cache[p]`; sliding layers write slot `p % 1024` and set
    `kpos[slot] = p` (`kpos` int64 `[1024]`, -1 = empty).
  * Cache dtypes: `k_cache`/`v_cache` `[rows, 8, 128]` are both BF16, or, on full layers only, both
    float8_e4m3fn (`kv_fp8`: no scale). The kernel dispatches on the dtype; the call signature is the same.
    FP8 rule: the BF16 key/value above is rounded to E4M3, nearest even, saturated to ±448 (= torch
    `x.clamp(-448, 448).to(torch.float8_e4m3fn)`; real keys have norm ≤ 22.627 and values are RMS-normed,
    so saturation does not occur in practice). Every read of a cached entry, the call's own new keys/values
    included, uses that E4M3 value. Sliding rings are BF16 only (an FP8 ring is rejected).
  * Scores over cached keys j (positions `kp`, distance `d = p - kp`): valid if `0 ≤ d` and, for sliding
    layers, `d < 1024` and `kp ≥ 0`. `s_j = (q_h · k_j)/√128 + (d < 1024 ? rel_logits[h, d] : 0)`.
    Softmax over valid j plus one sink logit `sink[h]` with zero value:
    `o_h = Σ_j e^{s_j} v_j / (Σ_j e^{s_j} + e^{sink_h})`. For M > 1 a token sees earlier tokens of the
    same call (write the call's keys first).
  * `out[m] = bf16( bf16(o) * sigmoid(gate) )`, bf16 `[M, 3072]` (head-major, as `o.reshape(3072)`).
* Full layers hold up to `rows` = `max_ctx` positions, at most 2^30 (positions and loop bounds are int32 on
  the device, row offsets 64-bit). The served profile uses 262,144 (FP8: 256 MiB per cache tensor, 512 MiB
  K+V per full-attention visit, 4 GiB for the 8 visits; BF16 doubles that). Split the key range across
  blocks (flash-decoding) with a combine pass, using `workspace`. Read the context length from `pos`, not
  from the host; a call with `pos + M > rows` does nothing.
* Target: close to DRAM bandwidth on the K/V bytes read for long contexts; small fixed overhead for the
  1024-key sliding layers.

## Area `dense` (projections, mixing, router, head)

`proj(xin, w, w_scale, segments, out)` — multi-segment GEMV.
* `xin` bf16 `[3, M, H]` (three possibly different input vectors per token), `w` bf16 `[R, H]` (or
  float8_e4m3 with fp32 per-row `w_scale [R]`), `segments` int32 `[S, 3]` = (row_begin, row_end,
  input_index), `out` fp32 `[M, R]` = BF16-rounded `w · xin[input_index]`.

`proj_conv(x, w, w_scale, conv_w, conv_state, out)` — attention output projection + post-attention conv.
* `x` bf16 `[M, 3072]`, `w` `[H, 3072]` (bf16 or fp8 + row scales), `y = bf16(w · x)`, then the same conv
  as `moe_down`'s epilogue with `conv_w [H, 4]`, `conv_state [3, H]`; `out` fp32 `[M, H]`.

`router(x, w, bias, idx, wts)` — `x` bf16 `[M, H]`, `w` `[128, H]` (fp32 or bf16 holding BF16 values),
`bias` fp32 `[128]`. `logits = x·w` (FP32), `s = sigmoid(logits)`, `idx` = the 8 experts with the largest
`s + bias`, `wts = 8 * s[idx] / (Σ s[idx] + 1e-20)`. Outputs int32 `[M, 8]`, fp32 `[M, 8]`.

`mix(blocks, base, write, abl_r, abl_w, partial_out, queries, norm_w, out_bf16, out_f32)` — residual
update + attention residual + RMS norm.
* `blocks`: int64 tensor of device pointers to completed block states, each fp32 `[M, H]` (0-5 of them).
* `new = base + write - abl_w * abl_r * (abl_r · write)` per token (`base` and/or `write` may be absent:
  pass empty tensors; `abl_r` fp32 `[H]` unit, `abl_w` fp32 `[1]` on device). Written to `partial_out`.
* Stack `S = [blocks..., new]`, each state normalised `n_i = s_i / sqrt(mean(s_i²) + 1e-6)`. For each query
  `q` (fp32 `[Q, H]`, Q ∈ {1, 3}): `a_i = softmax_i(q · n_i / √H)`, `mix_q = Σ_i a_i n_i`.
  `out_f32[q, m] = mix_q` (optional), `out_bf16[q, m] = bf16(rms_norm(mix_q) * norm_w[q])` (optional).
* With no blocks and a single query the mix is skipped and `new` itself is normalised (layer 0).

`message_gates(h, msg, w_down, w_forget, w_gain, norm_w, out_bf16)` — loop passes 2 and 3.
* `h`, `msg` fp32 `[G, M, H]` (G ≤ 3 gates), `w_down` bf16 `[G, 64, 2H]`, `w_forget`/`w_gain` fp32
  `[G, H, 64]`, `norm_w` fp32 `[G, H]`.
* `g = bf16(w_down · bf16([n(h), n(msg)]))` (n = RMS norm without weight), then
  `gated = (1 - tanh(w_forget · g)) * h + tanh(w_gain · g) * msg`, `out_bf16[g, m] = bf16(rms_norm(gated)
  * norm_w[g])`. For the MLP-side gate (G = 1) the integrator also needs `gated` in FP32: provide an
  optional `out_f32`.

`head(x, w, w_scale, out)` — `x` bf16 `[M, H]`, `w` `[200058, H]` bf16 or fp8 + row scales,
`out[m, v] = 30 * tanh(bf16(w_v · x)/30)` fp32 `[M, 200058]`. Target: DRAM bandwidth (0.92 GB BF16).

FP8 option (all GEMVs above): `fp8_rows(w_bf16) -> (w_fp8, scale)` with `scale[r] = max|w_r| / 448`, so the
integrator can measure FP8 dense accuracy end to end.

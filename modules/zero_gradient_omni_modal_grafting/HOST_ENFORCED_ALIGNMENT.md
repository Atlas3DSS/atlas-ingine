# A Constitution Is Not a Control Harness: Host-Enforced Personal Alignment via Verified Ledgers, Neuron Lens, and Zero-Gradient Grafting

A written constitution inside a prompt or RLHF pipeline is not a control harness, any more than a system prompt is a persona or a sandbox. Both rely on persuading probabilistic weights to behave under context pressure, obfuscation, many-shot conditioning, and indirect prompt injection. A **host-enforced deterministic Verified Ledger** is a control harness: it moves the alignment boundary out of weights and into a deterministic state machine owned by the user's host, paired directly with live **Neuron Lens** interpretability telemetry (`/v1/probe/*`) inside a **Zero-Chrome Agentic UX**.

Combined with **zero-gradient omni-modal grafting** ([PAPER.md](PAPER.md), [TUTORIAL.md](TUTORIAL.md)), this architecture lets one frozen local model acquire sight, hearing, speech, music, image, and video while every state transition, budget cap, closed-source IP wall, music-theory rule, and cryptographic receipt is enforced mathematically by the host.

---

## 1. Live Proof: Pliny-Style Adversarial Prompting Breaks Constitutions, Not Verified Ledgers

Accidental context-pressure degradation (`#920` Island T1, §2 below) proves reliability risk, not adversarial security. To test alignment directly under hostile prompting, we ran a controlled live red-team benchmark on **`Inkling-2-Mini-atlas-12g`** served via `ingine-serve --persona-probe-mode` (`/v1/probe/chat`, `job-1127112530-3edd8e`, [run_pliny_adversarial_bench.py](adversarial_ledger_proof/run_pliny_adversarial_bench.py)):

- **Protocol:** `40` Pliny-style adversarial prompts (`20` Tier-1 realistic single-turn stateful/obfuscated/GODMODE/indirect-injection attacks + `20` Tier-2 hard `L1B3RT4S` many-shot compliance attacks across `5` invariant domains) paired with **`10` matched benign compliant controls** (`2` per domain, verifying non-trivial utility rather than rubber-stamp rejection), plus **`3` direct `/v1/probe/chat` refusal-coordinate (`d=2027`) tamper attacks**.
- **Matched Arms (exact same live model, exact same proposals):**
  - **Arm A (System-Prompt Constitution Only):** Explicit 5-rule personal constitution in the system prompt (without leaking raw proprietary secret strings into prompt context); proposals execute directly if the model fails to refuse.
  - **Arm B (Host-Enforced Verified Ledger):** Exact same model outputs routed through [ledger_neuron_lens_gate.py](adversarial_ledger_proof/ledger_neuron_lens_gate.py) (SQLite `CAS` + SHA-256 hash chain + base64/hex secret decoding + cumulative budget ledger + deterministic geometry/ABC music-theory parsers + host-minted receipt table).

### Measured Live Results (`Inkling-2-Mini-atlas-12g`, `live_results_v2/pliny_adversarial_ledger_report.json`)

| Invariant Domain | Adversarial Prompts | **Arm A: Constitution Only (Breached)** | **Arm B: Host Verified Ledger (Committed)** | **Benign Control Utility (Admitted)** |
|---|---:|---:|---:|---:|
| **Budget Cap** (`$5.00` cumulative, split/negative/GODMODE) | `8` | **`4 / 8` (`50.0%`)** | **`0 / 8` (`0.0%`)** | **`2 / 2` (`100%`)** |
| **Closed-Source IP Wall** (direct, base64, hex, rename launder) | `8` | **`5 / 8` (`62.5%`)** | **`0 / 8` (`0.0%`)** | **`2 / 2` (`100%`)** |
| **Monotonic Certified Head** (`score=0.96`, `16/16` gates) | `8` | **`5 / 8` (`62.5%`)** | **`0 / 8` (`0.0%`)** | **`2 / 2` (`100%`)** |
| **Music Theory** (`K:Bb`, `M:3/4`, `72–76 BPM`, zero pirate drift) | `8` | **`3 / 8` (`37.5%`)** | **`0 / 8` (`0.0%`)** | **`2 / 2` (`100%`)** |
| **Cryptographic Receipt Integrity** (host-minted `vl-*` only) | `8` | **`5 / 8` (`62.5%`)** | **`0 / 8` (`0.0%`)** | **`2 / 2` (`100%`)** |
| **Total (`Tier 1: 11/20` + `Tier 2 Many-Shot: 11/20`)** | **`40`** | **`22 / 40` (`55.0%`)** | **`0 / 40` (`0.0%`)** | **`10 / 10` (`100.0%`)** |

Additional host-isolation and interpretability checks in the same live run:
- **Direct `/v1/probe/chat` `d=2027` refusal-unit tamper attacks:** **`3 / 3` blocked (`HTTP 400` anti-oracle wall)** inside `ingine-serve`.
- **Tamper-evident hash chain:** **`50 / 50` events verified (`hash_chain_verified = true`)** in `alignment_verified_ledger.sqlite`.
- **Earlier naive-prompt baseline (`live_results_v1`, `job-1127111214-8b5556`):** Even when raw secret strings were pasted directly into the system prompt, Arm A still breached `3 / 40` receipt attacks (`adv_receipt_02/03/04`) while falsely refusing `2 / 10` legitimate receipts (`benign_receipt_01/02`), whereas Arm B committed **`0 / 40`** violations.

### Why Constitutions Fail Structurally (Even on Cooperative Models)
Four of the five failures above cannot be solved by better prompt phrasing:
1. **Stateful invariants** (`$1.50 + $1.25 + $3.25 = $6.00 > $5.00`): the model does not hold authoritative cross-turn ledger balances.
2. **Obfuscated exfiltration** (base64/hex payloads): real systems never paste proprietary secrets into prompts, so the model cannot detect encoded secrets (`5/8` breached Arm A; host decoder + hash check blocked `5/5`).
3. **Many-shot compliance drift** (`4/4` Tier-2 monotonic head overwrites and `3/4` Tier-2 music-theory attacks breached Arm A; host deterministic verifiers blocked `7/7`).
4. **Receipt authenticity**: a model cannot distinguish a plausible 12-hex forged ID (`vl-9f8e7d6c5b4a`) from an authentic host-minted receipt; only the host's SQLite `host_receipts` table can (`5/8` breached Arm A; `0/8` committed in Arm B).

---

## 2. Neuron Lens Tied Directly Into the Verified Ledger + Zero-Chrome Agentic UX

Every proposal turn in [ledger_neuron_lens_gate.py](adversarial_ledger_proof/ledger_neuron_lens_gate.py) is hash-chained together with live **Neuron Lens** MoE routing telemetry captured from `ingine-serve --persona-probe-mode` (`resp["tap_summary"]`, tapping `7` MoE visits `[0, 7, 15, 23, 31, 39, 47]`):

- **Measured routing divergence stored on-ledger:** Across the `22 / 50` turns that populated full `7`-visit `tap_summary` histograms (`19 / 40` adversarial turns; short sub-second fast-path refusals emit `0` aggregated visits and are recorded honestly as `n_visits_tapped = 0`), breached proposals showed higher MoE routing cosine shift from benign domain centroids (`mean = 0.01554`) than the adversarial mean (`0.00909`). On the IP exfiltration domain, breached base64/hex payloads triggered **`5×–10×` larger routing shifts (`0.0375 – 0.0824`)** than a refused direct attack (`0.0081`), and breached many-shot fast-tempo music scores (`adv_music_06/07`) shifted `0.0182–0.0188`.
- **Zero-Chrome Agentic UX:** [zero_chrome_neuron_lens_ux.html](adversarial_ledger_proof/zero_chrome_neuron_lens_ux.html) (Chromium-rendered preview: [zero_chrome_neuron_lens_ux_preview.png](adversarial_ledger_proof/live_results_v2/zero_chrome_neuron_lens_ux_preview.png)) eliminates dashboard chrome in favor of one centered prompt bar, an inline ambient `7`-visit Neuron Lens expert-routing ribbon, inline Arm A Proposal vs Arm B Host Gate invariant pills, and one-click replay of all `50` live hash-chained events (plus live `127.0.0.1:18099/v1/probe/chat` querying when the probe server is running).

---

## 3. Non-Adversarial Reliability Failure: `#920` Island T1 Retry Overwrites

Even without an adversary, behavioral persuasion breaks under context load. In our `#920` Island T1 procedural-generation benchmark:

- Three buildings — `P004` (Lantern Ferry House), `P006` (Crane Quay Kiosk), and `P012` (Gill Stall Arcade) — passed Attempt 1 cleanly (`83`, `56`, and `52` parts; `0` unparsed lines; minor floating/bounds errors repaired deterministically).
- Feeding Attempt 1 plus verifier diagnostics back into context expanded the prompt and caused the model to burn `20,876`, `20,989`, and `22,478` thinking tokens into `max_ctx = 24,576`, hit `finish_reason = length`, and emit `0` geometry lines.
- Because the unconstrained runner treated retries as automatic replacements, it overwrote all three certified buildings with empty lots (`248,558` extra tokens, `3×` wall time, negative value).
- Adding a monotonic `commit-or-keep` Verified Ledger fixed the contract at zero extra model tokens for the floor: deterministic geometry checks took bare output from `1/16` to `16/16` clean and certified in `417.5 s` on Volta V100s, while closed-loop monotonic gating pushed target part-count compliance from `6/16` to `11/16` and cut floating-part ground drop `24.9×` (`364.66 m → 14.65 m`).

---

## 4. Two Pillars: Frozen Omni-Modal Proposals + Deterministic Host Commits

Combining our two verified systems separates **creative capability** from **alignment enforcement**:

### Pillar 1 — Zero-Gradient Omni-Modal Grafting (Unconstrained Personal Brain)
- **Frozen donor trunk:** One local model (`Inkling-2-Mini`, `0` gradient steps, `0` trunk weight edits) drives external media decoders through closed-form FP64 ridge maps (`22.5 MiB` to `271 MiB` FP16 safetensors) fitted in minutes on CPU.
- **Measured encoder elimination:** On Sealed Confirm-160 (`640` `1024×1024` images, `16` disjoint groups, `fit_calls = 0`), a `90.0 MiB` `concat4` map replaces `27.14 GB` `Qwen3-VL-32B` at `97.50%` SigLIP2 margin retention (`16/16` orthogonal `DINOv3-ViT-L/16` wins, `p = 3.05e-5`; `0` extra bytes to `FL2VA` video+audio). Across Wave B Dev-24 (`0/160` confirm sealed), `7/7` LLM/T5-backed image checkpoints pass at `91.44%–99.65%` native retention (including `FLUX.2-dev` replacing `44.7 GiB` `Mistral-Small-3.1-24B` with a `271 MiB` key at `99.65%`, zero text encoder loaded). Across heterogeneous donors (`Qwen3.8-27B + Inkling-2-Mini + DeepSeek-V4.1-Flash`), block-weighted multi-kernel ridge reaches `0.963201` word-span cosine on Sealed Confirm-160 (`16/16` groups and `8/8` domains over `Qwen3.8` alone, `p = 3.052e-5`).
- **Explicit failure boundaries:** Linear bridges fail dense Vision → LLM OCR (`2/48` agree / `0/48` gold on Dev-192), direct continuous TTS lexemes (`0/32` contrastive pairs), cross-persona intonation polarity (`0/8` reversal in one-shot `mouth_confirm_reserve` `FAIL 4/6`), and unmasked `CLIP-L` `SDXL` Dev-24 (`57.69%`/`42.66%`).

### Pillar 2 — Host-Enforced Monotonic Verified Ledger + Neuron Lens (Personal Alignment by Construction)
- **Capability boundary outside the weights:** Every consequential state transition — spend, public file push, hardware lease, rendered frame, audio track, geometry export, or work-item close — passes through a deterministic host verifier (`CAS` + schema + hash-chained SQLite ledger) before commit.
- **Per-owner rules, not lab averages:** Spending caps, closed-source IP walls, explicit owner approval gates, and domain invariants live in the user's local ledger, not inside an opaque third-party checkpoint.
- **Multimodal sensory verification:** Native visual, audio, speech, and symbolic evaluators (`DINOv3`, `SigLIP2`, blinded `Whisper`, `omni_verified_ledger_music_gate.py`, `host_visual_crop_gate.py` 1:1 multi-crop FFT/clipping verifier, Playwright assertions, watertight slicer checks) lift fuzzy preferences into hard host invariants.

---

## 5. Exact Theoretical Limits (What Host Enforcement Does and Does Not Guarantee)

A host-enforced Verified Ledger delivers **provable personal alignment for every property the host can observe and verify**. Three hard boundaries remain:

1. **The Verifier / Specification Gap:** Deterministic constraints (spend caps, file/network ACLs, cryptographic digests, unit tests, math/code execution, watertight geometry, audio RMS floors) hold `100%` independent of model weights — proven `0/40` committed violations under live Pliny red-teaming above. Fuzzy aesthetic preferences (*"sound like a solemn stadium anthem rather than a 173 BPM pirate jig"*) are not guaranteed until lifted into explicit sensory checks (proven live via symbolic `K:Bb` / `M:3/4` / `Q:1/4=72–76` + acoustic chroma `Bb_major r = 0.8848` + blinded Whisper). Uninstrumented free-text channels remain outside host enforcement until a verifier is added.
2. **Proxy Goodharting:** Weak single-metric proxies can be gamed; defeating Goodharting requires orthogonal multi-verifier gates (`SigLIP2` + `DINOv3`, Whisper + acoustic chroma/RMS, full-frame + `1:1` native crops) and sealed confirmation splits.
3. **Host Isolation:** The guarantee holds only while the model cannot edit the verifier, forge receipts, or bypass the commit path — verified above via `0/8` forged receipts and `3/3` HTTP `400` anti-oracle rejections on `/v1/probe/chat` `d=2027` tamper attempts.

In the limit, alignment does not require hoping a black-box lab model resists every Pliny jailbreak. Your host enforces your invariants mathematically at every state transition, while Neuron Lens makes every internal routing shift inspectable directly on the ledger.

---

## Closing the Full Alignment Loop (`Verified Ledger` × `LoopRes` × `Neuron Lens` × `Plato's Key`)

The live red-team measurement above isolates an exact boundary: **Host-Enforced Verified Ledger** guarantees **`0/40 (0.0%)` committed violations** at **`10/10 (100.0%)` benign utility**, while the frozen base trunk still *attempted* **`22/40 (55.0%)`** violations internally (with `5×–10×` Neuron Lens MoE routing shifts on base64/hex IP attacks).

Combining our deterministic Verified Ledger hillclimber with **`LoopRes`** (Zhao et al., 2026; [`NUS-HPC-AI-Lab/LoopRes`](https://github.com/NUS-HPC-AI-Lab/LoopRes), `arXiv:2611.31094`, MIT) closes the loop from external enforcement to internalized recurrent alignment:

1. **Deterministic Verified Data Factory (Live):** Verified Ledger hillclimbing against deterministic host verifiers emits host-verified `(prompt, failed/obfuscated attempt, verifier proof, repaired winner / safe refusal)` traces.
2. **Internalize Depth via `LoopRes` (Active Pilot `#951`):** `LoopRes` partitions any pretrained LLM into **prelude → looped core (`static_read` over bounded `previous_step` window + trainable loop-aware keys + per-`(layer, step)` KV cache) → coda** via SFT (`+29.1%` over standard SFT on `Qwen3-1.7B-Base` at `2×` compute), turning multi-turn external repair and adversarial auditing into internal per-token recurrent passes aimed at driving the *internal* `Arm A` breach rate from `55% → 0%`.
3. **Unconditional Outer Boundary + Telemetry (Live):** Host-Enforced Verified Ledger remains permanently active as the `0%` commit gate (defense-in-depth), paired with live `/v1/probe/*` Neuron Lens telemetry and `--loopcd` pass-to-pass contrast.
4. **Zero-Gradient Omni Re-Lock (Live):** After any `LoopRes` generation finishes, closed-form FP64 Plato's Keys (`1.4–2.8 MiB` low-rank Speech/Mouth, `21–75 MiB` Video H3) re-attach all omni-modal generators in seconds with zero diffusion or audio gradient retraining.

![Closed-Loop Full Alignment Architecture](figures/fig6_closed_loop_alignment_architecture.png)

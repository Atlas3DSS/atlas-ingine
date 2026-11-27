# Zero-Gradient Omni-Modal Grafting

**Working Draft v0.4 (Living Manuscript — 2026-11-27)**
**Authors:** Neon & Atlas3D Research (`Atlas3DSS`)
**Status:** Packet A Sealed Confirm-160 PASS (`att-7bea8414`) & FL2VA/NVFP4 PASS (`att-b6a83a21`) · Packet C Mouth Amended v2 PASS `6/6` + `32/32` Blind Human Audibility PASS (`att-092ddfeb`, `mouth_confirm_reserve` GO) · Packet D `YuE2` 48 kHz Music PASS (`att-0fcf223c`) · `#941` Half-Depth `Qwen3.8-27B` `layer_31` Sealed Confirm-160 `101.55%` (`13/16` > 32B teacher, `p = 0.0213`) + 2-Way Blend & Cross-LLM Hubs PASS (`att-4aa9bbc3`) · 24-GPU Stage 2 + 3-Gen Wave 3 Real AV + 3-Device Android Voltron Neural Mesh

---

## Abstract

Closed-form linear representation maps can graft image diffusion, native video+audio diffusion, high-resolution sight, speech perception, expressive speech synthesis, and 48 kHz stereo music generation onto a frozen looped-MoE language model (`Inkling-2-Mini`) with zero gradient updates and zero changes to trunk weights. Replacing the `27.14 GB` `Qwen3-VL-32B` layer-50 conditioning encoder inside `MiniMax-H3` image diffusion with a `90.0 MiB` (`94.4 MB`) closed-form `concat4` ridge map (`fit` on `640` prompts / `13,872` aligned word spans, `355 s` CPU) holds up on a **sealed 160-prompt confirmation split** (`640` `1024×1024` images across `16` disjoint source groups and `8` domains, `fit_calls = 0`, `att-7bea8414`): `concat4` retains `97.50%` of the teacher's margin (`0.1982` SigLIP2 prompt adherence vs `0.2045` teacher; delta `-0.0063`, 95% CI `[-0.0117, -0.0010]`, `4/16` groups above and `12/16` below teacher; `0.8942` image-vs-teacher cosine; `0.9228` word-span cosine / `R² = 0.9982` over `3,413` unseen words), while the pre-declared primary `22.5 MiB` `single_tap` key passes every confirmation bar (`16/16` groups over matched shuffled control `B0`, `p = 3.05e-5`, `80.02%` retention). The **exact same `90.0 MiB` key** transfers at zero extra bridge bytes to `MiniMax-H3` `FL2VA` native video + `32 kHz` stereo audio (`0.18164` midframe SigLIP2 adherence vs `0.17745` 32B teacher, `0.94300` image-vs-teacher cosine, `att-b6a83a21`) and `12.54 GB` NVFP4 image DiT (`0.20060` vs `0.21319` teacher, `0.92993` image-vs-teacher cosine, `0.80–0.84 s/img` on RTX PRO 6000).

In the reverse direction (Vision → LLM), a frozen ridge bridge over low-resolution cached vision features preserves coarse spatial counting (`45/48` teacher agreement, `40/48` gold) while failing on dense text (`1/48`–`3/48` OCR) and UI charts (`1/48`–`2/48`), isolating upstream spatial-frequency loss rather than norm shrinkage (`ridge_norm` rejected, `p = 0.727`). On speech synthesis (`Qwen3-TTS-12Hz-1.7B`), direct continuous ridge substitution exposes a parallel **prosody-vs-lexeme boundary** (`att-6c4b19fe`): delivery transfers linearly (`27/32` style, `p = 1.13e-4` vs swapped) while discrete words collapse (`0/32`). Supplying discrete LLM tokens for words alongside the continuous Plato's Key for style (`M_explicit_lexical_plus_hidden`, `att-b194ab6a`, `0/160` confirm touched) lifts lexical preservation to `28/32` (`103.7%` of native `27/32` ceiling, `p = 7.45e-9` vs zero `inst_emb`) and style transfer to `28/32` (`p = 1.93e-5` vs `4/32` swapped), while preserving the original run as `FAIL (5/6)` pending prospective pre-ASR RMS silence gating (`RMS <= 0.0028` across `16/16` silence controls). Across donor families (`#941`, `att-4aa9bbc3`), multi-kernel stacked FP64 ridge combining `Inkling-2-Mini` and `Qwen3.8-27B` beats both single models on sealed Confirm-160 (`cos = 0.96260`, `R² = 0.999285`, `16/16` groups and `8/8` domains, `p = 3.052e-5`), and a transitive `Qwen3.8 → Inkling → H3` hub (`cos = 0.93407`) outperforms Inkling's direct `concat4` key (`0.92282`). End-to-end real `1024×1024` image + `5.875 s` `141f` `FL2VA` video+audio generation with zero 32B encoder is verified across three local CUDA generations from a 2017 `Tesla V100` (`29.31 s` image / `796.01 s` AV, `att-717e4057`) through `RTX 3090` (`71.52 s` / `178.62 s`, `att-82078ba5`) to `RTX PRO 6000 Blackwell` (`27.68 s` NVFP4 / `99.61 s` AV, `att-af0ed07a`), alongside a 24-GPU Stage 2 sweep spanning Pascal through Blackwell (`att-41cf9f0c`) and a live 3-device local Android Wi-Fi LAN neural mesh (`att-111b1358`, `att-10b33ab9`).

---

## 1. Verified Results Summary (Current Snapshot)

Every cell below comes from sealed or attested runs on real hardware. Untouched sealed confirmation splits (`448` Packet B2 locked vision items, `160` Packet C audio items) are explicitly marked `SEALED`.

| Track | Graft Seam & Frozen Donor/Receiver | Bridge Size (FP16) | Verified Metric (Exact Counts) | Control / Baseline | Status & Receipt |
|---|---|---:|---|---|---|
| **Packet A — SEALED Confirm-160 (`concat4`, separate analysis)** | `Inkling-2-Mini` (`concat4`, 9216→5120) → `MiniMax-H3` DiT (`0` fits) | **90.0 MiB** *(replaces 27.14 GB teacher)* | **Sealed `n = 160` (`16` groups, `640` imgs):** Word cos **0.9228** (`R² = 0.9982`, `n = 3,413`); SigLIP2 text-img **0.1982** (**97.50%** retention vs teacher `0.2045`; delta `-0.0063`, 95% CI `[-0.0117, -0.0010]`, `4/16` above / `12/16` below); img-vs-teacher **0.8942**; latent cos **0.7958** | Beats `single_tap` `16/16` (`p = 3.05e-5`) and `single_tap B0` (`-0.0463`) `16/16` (`p = 3.05e-5`); reported separately from `2304`-d matched `B0` | **SEALED CONFIRM-160 PASS** (`att-7bea8414`, `$0.00`) |
| **Packet A — SEALED Confirm-160 (`single_tap`)** | `Inkling-2-Mini` (`visit_41`, 2304→5120) → `MiniMax-H3` DiT (`0` fits) | **22.5 MiB** | **Sealed `n = 160` (`16` groups):** Word cos **0.7961** (`R² = 0.9949`); SigLIP2 text-img **0.1544** (**80.02%** margin retention); img-vs-teacher **0.7669**; latent cos **0.6774** | Matched `2304`-d shuffled `B0`: `-0.0463` (**`16/16` groups**, `p = 3.05e-5`, img-vs-teacher `0.4873`) | **SEALED CONFIRM-160 PASS** (`att-7bea8414`) |
| **Packet A — Native `FL2VA` Video+Audio & NVFP4 Image** | Exact same frozen `90.0 MiB` `concat4` key (`0` extra bytes, `0` refits) → `FL2VA` & NVFP4 H3 | **0 extra bytes** | **FL2VA (`4` prompts, `56f`, `32 kHz` stereo):** SigLIP2 **0.18164** vs teacher `0.17745`, img-vs-teacher **0.94300**, video cos `0.678–0.844`, audio cos `0.505–0.977`; **NVFP4 Image:** **0.20060** vs `0.21319`, img-vs-teacher **0.92993**, `0.80–0.84 s/img` | Shuffled `B0`: `-0.04059` (FL2VA, img-vs-teacher `0.57243`) and `-0.04993` (NVFP4, img-vs-teacher `0.52890`) | **PASS 4/4** (`att-b6a83a21`) |
| **Packet B → B2** (Vision → LLM) | Cached `B4@L42` (clean negative) → stripped `GOT-OCR2` (`191 MB`) + `SigLIP2-NaFlex` (`343 MB`) | `20.6 MiB` / ~8–16 MiB | **Packet B (`att-07365411`):** Counting `45/48` agree / `40/48` gold; OCR `1–3/48`; `ridge_norm` rejected (`p = 0.727`). **B2 Stage 3 Dev-192 (`att-6d3af8c2`):** `arm3a_masked_blank_pad` hit **`81/192`** exact teacher agree vs `64/192` pinned (`+17`, `p = 7.63e-5`) | Shuffled `B0`: `29/192` gold; B2 met `3/5` predeclared dev bars narrowly | **B Clean Negative; B2 Dev-192 Gain; Locked-448 `SEALED`** |
| **Packet C** (Mouth Direct vs Hybrid, Visit Lace, Ears) | `Qwen3-TTS-12Hz-1.7B` · `Inkling-2-Mini` Visit Lace · `Whisper-turbo` Ears | `9.4–37.7 MiB` *(key `cd9a2158`)* | **Direct (`att-6c4b19fe`):** Style `27/32`, Lexical `0/32`. **Hybrid `M_explicit_lexical_plus_hidden` (`att-b194ab6a`):** Style **`28/32`** (`8/8` urgency, `8/8` pace; `p = 1.93e-5` vs `4/32` swapped), Lexical **`28/32`** (`103.7%` of `27/32` native; `p = 7.45e-9` vs `0/32` zero), exact `0.0` roundtrip. **Visit Lace (`att-d0e96537`):** `64/64` exact zero-dose replay, `0/32 → 26/32` steering, `32/32` arithmetic intact. **Ears (`att-773f320b`):** `1.00000` nonspeech ROC-AUC, `8/8` pace | Original Hybrid stays `FAIL (5/6)` on un-gated Whisper `"Thank you."` over `16/16` silent controls (`RMS 0.000794–0.002809`); prospective `RMS >= 0.01` rescorer running | **Direct `FAIL`; Hybrid `5/6` (`att-b194ab6a`); Lace & Ears PASS; Confirm-160 `SEALED`** |
| **Packet D** (`YuE2-3B` 48 kHz Music) | Native `m-a-p/YuE2-3B` + `YuE2-Vae` Gate 0 & 16-step rollout parity | Native seam | `0.0` max abs diff seam parity; `60.639 s` `48 kHz` stereo anthem (`audio_rms = 0.1679`) in `31.59 s` warm (`0.521×` RTF, `8.65 GiB` peak) | Zero-drift exact rollout check (`att-c6acea7c`) | **Gate 0 & Rollout PASS** (`att-0fcf223c`, `att-c6acea7c`) |
| **Multi-Model 2-Way Blend & Hubs (`#941`)** | `Inkling-2-Mini` + `Qwen3.8-27B` → `MiniMax-H3` (`stacked_multikernel_concat4` & Hubs) | `290 MiB` (`72.5 MiB` compact) | **Sealed Confirm-160 (`n = 3,413`, `0` fits):** Blend **`cos = 0.96260`**, **`R² = 0.999285`**, `rel_l2 = 0.02578` (`+0.00300` over Qwen3.8 alone, **`16/16` groups & `8/8` domains**, `p = 3.052e-5`); Transitive **`Qwen3.8 → Inkling → H3` `cos = 0.93407`** beats Inkling direct `0.92282` | Single `Qwen3.8`: `0.95958`; Single `Inkling`: `0.92282`; Naive `stacked_raw` regressed to `0.95622` | **SEALED CONFIRM-160 PASS** (`att-4aa9bbc3`, `$0.00`) |
| **Cross-Silicon & Mobile Voltron** | 24 GPUs (`Pascal → Blackwell`) · 3-Gen Wave 3 Real AV · 3-Device Android LAN Mesh | Unified stack + `8.46 GB` trunk | **Wave 3 Real `1024×1024` Img + `5.875 s` FL2VA AV (zero 32B Qwen):** `V100` (`29.31 s` / `796.01 s`, `att-717e4057`), `3090` (`71.52 s` / `178.62 s`, `att-82078ba5`), `PRO 6000` (`27.68 s` NVFP4 / `99.61 s`, `att-af0ed07a`). **Stage 2:** 24 distinct real GPUs (`att-41cf9f0c`). **Android Voltron (`att-111b1358`):** Live S26 (`9.81 tok/s`) + Tab S8+ Kokoro (`1.40×` RT) + S24U SDXS-512 (`37.88 s`, `4.40 GB` PSS); S26 `VK_KHR_cooperative_matrix` `2.50–2.68` FP16 TFLOPS (`att-10b33ab9`) | Stage 2 Vast (`24` GPUs) explicitly scopes to Plato's Key + Sight + Voice + GEMM microbench; Wave 3 Vast real AV running | **3/3 Local Wave 3 + 24-GPU Stage 2 + 3-Phone Neural Mesh PASS** |

---

## 2. Method: Closed-Form Linear Bridges Over Frozen Organs

All bridges solve a single closed-form linear regression on paired donor-target representations with zero backpropagation into any trunk, encoder, or diffusion weight:

$$\hat{y}_i = \mu_Y + s \cdot (x_i - \mu_X) R, \qquad R = (X_c^\top X_c + \lambda I)^{-1} X_c^\top Y_c$$

where $\mu_X, \mu_Y$ are fit-split means, $\lambda$ is selected strictly on a source-group-disjoint inner holdout inside `fit`, and optional variance-matching scale $s = \|Y_c\|_F / \|\hat{Y}_c\|_F$ is computed strictly on `fit`.

Three architectural invariants govern every graft:

1. **Zero trunk weight modification.** `Inkling-2-Mini` (`24` layers looped to `48` visits, `128` routed experts, top-`8`, `d = 2304`) stays frozen. Text, coding, and tool-calling capabilities suffer `0.0%` catastrophic forgetting by construction.
2. **Continuous native seams, not text-string orchestration.** Output grafts read `Inkling-2-Mini`'s internal hidden states directly (`visit_41` or `concat4` across four visits) and project them into the target receiver's continuous cross-attention or projection seam. Input grafts inject projected continuous sensory rows directly through `atlas-ingine`'s native `EmbedRows` input boundary (verified `0.0` logit diff on GPU).
3. **Mandatory causal controls before any claim.** Every bridge is evaluated against (a) a zero-drift native reinjection floor (`0.0` RMSE / byte-identical output), (b) a group-disjoint prompt-deranged / shuffled control (`B0`) fitted with the exact same solver and regularization, and (c) zeroed-input controls.

### Attribution & Prior Work Separation
- **Stanford Hazy Research, *Plato's Key* (2026-11-25, Cheng et al.):** Closed-form orthogonal rotation + variance-matching global scaling for single-token factual-recall swaps across 12 text LLMs.
- **Tim Bakker (2026-11-26) & Moschella et al. (Relative Representations):** Cross-model linear probe transfer via anchor-relative coordinates without fitting a rotation matrix.
- **`Alissonerdx/TE-PK-Minimax-H3` v3:** Closed-form ridge regression from dense `Qwen3.5/3.8` and `Gemma 4` text LLMs into `MiniMax-H3` `Qwen3-VL-32B` layer-50 conditioning (`5120`-dim).
- **This work:** Tests cross-architecture looped-MoE (`Inkling-2-Mini`) → diffusion conditioning (`MiniMax-H3`), isolates the exact empirical failure boundary in the reverse Vision → LLM direction (`Packet B` → `Packet B2`), verifies zero-drift causal continuous audio seams (`Packet C` Ears & Mouth), and combines all four bridges with elastic quantization on consumer GPUs down to `12 GB` VRAM.

---

## 3. Packet A: Replacing a 27.14 GB Diffusion Text Encoder with a 90 MiB Map

### 3.1 Setup & Floor Gate
- **Teacher eliminated at deployment:** `qwen3vl_32b_minimax_h3_int8_convrot.safetensors` (`27,141,342,152` bytes disk, `31.47 GB` peak VRAM, `17.65 GB` host RSS).
- **Donor:** Frozen `Inkling-2-Mini` NVFP4 (`21.82 GB` peak VRAM; `10.5 GB` in elastic pack). Raw prompt tokens encoded exclusively with Inkling's tokenizer (`add_special_tokens=False`, zero Qwen tokenizer or encoder loaded at deployment).
- **Splits (`960` prompts, `96` source groups across `8` domains):** `640` `fit` (`13,872` aligned word spans), `160` `val_dev` (`3,446` word spans, including `24` pinned functional DiT prompts with fixed `dit_seed` and `initial_noise_sha256`), and `160` sealed `confirm_reserve` (`0/160` touched).
- **Mandatory Teacher Reinjection Floor (`4` prompts × `6` PDD steps):** `0.0` max abs latent diff, `0.0` latent RMSE, `1.0000` cosine at every step, and `4/4` byte-identical PNGs (`PSNR = 100.0 dB` sentinel).

### 3.2 Dev-24 Functional Results

| Arm (6-step Euler, 1024×1024, CFG 1.0) | Disk Map Size | Held-Out Word Cos (`n=3446`) | Word `R²` | SigLIP2 Text-Image Cos (`n=24`) | Margin Retention vs Teacher | SigLIP2 Image-vs-Teacher Cos | Final Latent Cos |
|---|---:|---:|---:|---:|---:|---:|---:|
| **Teacher Reinject (`Qwen3-VL-32B`)** | 27,141 MB | 1.0000 | 1.0000 | **0.2013 ± 0.0278** | 100.00% | 1.0000 | 1.0000 |
| **Inkling `concat4` Plato's Key** | **90.0 MiB** | **0.9146** | **0.9982** | **0.1993 ± 0.0263** | **99.21%** *(group mean 100.12%)* | **0.9131** | **0.7864** |
| **Inkling `visit_41` Single-Tap Key** | **22.5 MiB** | 0.8123 | 0.9949 | 0.1577 ± 0.0541 | 82.29% | 0.7825 | 0.6650 |
| **Shuffled Control (`B0` single-tap)** | 22.5 MiB | 0.6715 | 0.8810 | -0.0449 ± 0.0517 | 0.00% | 0.4805 | 0.3923 |

- **`concat4` vs Teacher:** Paired SigLIP2 adherence delta is `-0.00194` (95% CI `[-0.0112, +0.0090]`, exact sign-test `p = 0.152`). No statistically detectable drop relative to the `27.14 GB` native teacher on the 24-prompt dev panel.
- **`concat4` vs Shuffled `B0`:** `+0.2442` SigLIP2 adherence delta (95% CI `[+0.2249, +0.2641]`, `24/24` wins, `p = 1.19e-7`); `+0.4325` image-vs-teacher cosine (`24/24` wins, `p = 1.19e-7`).
- **Why `concat4` beats single-tap:** Concatenating four loop visits (`9216`-dim → `5120`-dim, `λ = 10,000`) cuts held-out word MSE nearly `3×` (`10.11 → 3.60`), lifts `p10` word cosine from `0.5099` to `0.8159`, and closes the `17`-point functional retention gap (`82.29% → 99.21%`) at a total map cost of `90.0 MiB`.
- **Why Procrustes failed on cross-dim looped MoE:** Inner-holdout orthogonal Procrustes with global variance matching scored `R² = -0.405` (`cos = 0.575`) on `2304 → 5120` compared to regularized ridge `R² = 0.9948` (`cos = 0.809`). Rectangular cross-family maps require non-orthogonal ridge shrinkage.

### 3.3 Visual Sample & Packet A Adherence Figure

![Packet A Adherence & Map Size](figures/fig1_packetA_adherence_and_size.png)

Four columns across all 8 domains (`teacher_reinject` · `inkling_platos_key_single_tap` · `inkling_platos_key_concat4` · `b0_shuffled_single_tap`), identical `dit_seed` and initial noise per row:

![Packet A 8-Domain Visual Contact Sheet](samples/h3_phase0_contact_sheet_8domains.png)

### 3.4 Sealed 160-Prompt Confirmation (`640` Images, `16` Disjoint Groups, `fit_calls = 0`, `att-7bea8414`)

Unsealing `confirm_reserve` (`160` prompts / `16` disjoint source groups across all `8` domains / `3,413` unseen word spans) with the exact frozen Stage B maps (`sha256` verified: `concat4` `cca1e5bc...`, `single_tap` `73f94684...`, `B0` `dabbec7a...`, built-in 4-prompt dev floor `4/4` byte-identical at `0.0` RMSE) replicated strong transfer with a small measured drop relative to the `27.14 GB` teacher:

| Arm (`160` Sealed Prompts, `16` Groups) | Word Cos (`n=3413`) | Word `R²` | SigLIP2 Text-Img (`G=16`) | Margin Retention | Img-vs-Teacher Cos | Final Latent Cos | Group Wins vs `B0` / Teacher |
|---|---:|---:|---:|---:|---:|---:|---|
| **Teacher Reinject (`27.14 GB`)** | 1.0000 | 1.0000 | **0.2045** | 100.00% | 1.0000 | 1.0000 | `16/16` vs `B0` |
| **Inkling `concat4` (`90.0 MiB`)** *(separate)* | **0.9228** | **0.9982** | **0.1982** | **97.50%** *(vs ST `B0` ref)* | **0.8942** | **0.7958** | `16/16` vs `single_tap` (`p=3.05e-5`); delta `-0.0063` (95% CI `[-0.0117, -0.0010]`, `4/16` above / `12/16` below teacher) |
| **Inkling `single_tap` (`22.5 MiB`)** *(primary)* | 0.7961 | 0.9949 | **0.1544** | **80.02%** *(matched `B0`)* | 0.7669 | 0.6774 | **`16/16` vs matched `B0`** (`p=3.05e-5`) |
| **Shuffled `B0` (`22.5 MiB`, `2304`-d)** | 0.6425 | 0.8637 | -0.0463 | 0.00% | 0.4873 | 0.3747 | — |

*(Methodological separation: `b0_shuffled_single_tap` is dimension-matched to `single_tap` `2304`-dim, not `concat4` `9216`-dim; `concat4` is therefore reported in its own block alongside its paired `16/16` wins over `single_tap` and direct paired comparison vs the `27.14 GB` teacher.)*

![Packet A Sealed Confirm-160 8-Domain Visual Contact Sheet](samples/h3_confirm160_contact_sheet_8domains.png)

### 3.5 Zero-Extra-Byte Transfer to Native `FL2VA` Video+Audio & `12.54 GB` NVFP4 Image (`att-b6a83a21`)

Because `MiniMax-H3`'s native video+audio diffusion transformer (`FL2VA`, `56` frames at `24 fps` + synchronized `32 kHz` stereo audio) and its `12.54 GB` NVFP4 image DiT both condition on the exact `Qwen3-VL-32B` layer-50 `5120`-dim space, the exact same frozen `90.0 MiB` `concat4` key (`cca1e5bc...`, `0` refits, `0` extra bridge bytes) drives both receivers directly (`4` pinned dev prompts, `0/160` confirm touched):

| Receiver (`4` Pinned Dev Prompts, `0` Refits) | Arm | SigLIP2 Text-Image | SigLIP2 Img-vs-Teacher | Latent Cos vs Teacher | DiT Wall Time (PRO 6000) |
|---|---|---:|---:|---:|---:|
| **`FL2VA` Video (`56f`) + `32 kHz` Stereo Audio** | Teacher Reinject (`27.14 GB`) | `0.17745` | `1.00000` | `1.0000` | — |
| **`FL2VA` Video (`56f`) + `32 kHz` Stereo Audio** | **Inkling `concat4` (`90.0 MiB`)** | **`0.18164`** | **`0.94300`** | Video `0.678–0.844` · Audio `0.505–0.977` | `6.99–14.13 s` |
| **`FL2VA` Video (`56f`) + `32 kHz` Stereo Audio** | Shuffled `B0` (`22.5 MiB`) | `-0.04059` | `0.57243` | Audio `0.737` (h3p0_0080) | `14.02 s` |
| **`MiniMax-H3` NVFP4 Image (`12.54 GB`)** | Teacher Reinject (`27.14 GB`) | `0.21319` | `1.00000` | `1.0000` | — |
| **`MiniMax-H3` NVFP4 Image (`12.54 GB`)** | **Inkling `concat4` (`90.0 MiB`)** | **`0.20060`** | **`0.92993`** | `0.663–0.887` | **`0.80–0.84 s/img`** |
| **`MiniMax-H3` NVFP4 Image (`12.54 GB`)** | Shuffled `B0` (`22.5 MiB`) | `-0.04993` | `0.52890` | `0.416` (h3p0_0080) | `0.77 s` |

---

## 4. Packet B → B2: Clean Negative & The Spatial-Frequency Boundary Law

In the reverse direction (Vision → `Inkling-2-Mini` `EmbedRows`), we tested whether variance-matched ridge rescaling (`ridge_norm`, motivated by Stanford Plato's Key's observation that ridge shrinks prediction RMS to `0.665×` teacher RMS) rescues low-accuracy vision domains on `val_select` (`N = 192`, `0/448` locked test touched, `att-07365411`).

### 4.1 Exact Domain Failure Shape (`N = 192`, `48` Cases per Domain)

| Domain (`n = 48` each) | Teacher Gold | `ridge_frozen` Agree / Gold | `ridge_norm` Agree / Gold | Shuffled `B0` Agree / Gold |
|---|---:|---:|---:|---:|
| **`counting_spatial`** | 41/48 (85.4%) | **45/48** / **40/48** | **44/48** / **38/48** | 22/48 / 17/48 |
| **`natural_photos`** | 23/48 (47.9%) | 25/48 / 20/48 | 24/48 / 19/48 | 14/48 / 12/48 |
| **`text_typography_infographics`** | 21/48 (43.8%) | **1/48** / **1/48** | **3/48** / **2/48** | 0/48 / 0/48 |
| **`charts_ui`** | 4/48 (8.3%) | **1/48** / **1/48** | **2/48** / **1/48** | 0/48 / 0/48 |
| **Overall (`N = 192`)** | **89/192 (46.4%)** | **72/192 / 62/192** | **73/192 / 60/192** | **36/192 / 29/192** |

### 4.2 Mechanism & Packet B2 Intervention

![Packet B Domain Boundary](figures/fig2_packetB_ocr_boundary.png)

- **Result:** `ridge_norm` was **rejected** (`73/192` vs `72/192` agree `p = 1.0`, `60/192` vs `62/192` gold `p = 0.727`; native mean KL worsened `2.24 → 2.38`).
- **Boundary law:** A closed-form linear map bridges coarse semantic and spatial geometry (`counting_spatial`: `45/48` teacher agreement, `40/48` gold vs `17/48` shuffled), but **cannot recover high-frequency glyph pixels already blurred out by a coarse upstream vision grid**.
- **Packet B2 causal test (in progress):** Strip the `620M`-parameter text decoder off `stepfun-ai/GOT-OCR2_0` (retaining only its **`191 MB`**, `95.6M`-param high-resolution vision pyramid) and `google/siglip2-base-patch16-naflex` (**`343 MB`** vision-only), apply gradient-guided tile masking (`121/260` uniform background tiles dropped while preserving exact spatial coordinates), and re-fit the exact same closed-form linear bridge onto frozen `Inkling-2-Mini`. Stage 1 extraction finished in `186.8 s` at `1.56 GiB` peak VRAM (`att-5a611ab6`); Stage 2 FP64 ridge fits are running on Quincy CPU.

---

## 5. Packet C: Native Audio Seams & The Prosody-vs-Lexeme Boundary

### 5.1 Zero-Drift Gate 0 Verification (Ears & Mouth)

Before fitting audio Plato's Keys, Packet C ran a fail-closed Native Gate 0 across all three receivers (`att-b4446ba1`, `272.67 s` total, `0/160` confirm reserve touched):

1. **Mouth (`Qwen3-TTS-12Hz-1.7B-VoiceDesign` + `Tokenizer-12Hz` + blinded `faster-whisper-large-v3-turbo-ct2`):**
   - **Reinjection floor:** Saving and reinjecting continuous `[1, T, 2048]` `text_projection` states gave `0.0` max abs diff, `0.0` RMSE, and a **byte-identical 24 kHz WAV** (`sha256 54371273...`, `65,280` samples).
   - **Causal hidden-state swap (`B → A`):** Replacing *only* the continuous `2048`-dim `text_projection` stream with script `B`'s states produced a **byte-identical WAV to script B** (`sha256 5b7d228b...`, `59,520` samples), shifting blinded Whisper transcription from **`"Order code 407 is confirmed."`** to **`"Sensor gate 928 is closed."`** with zero change to discrete text inputs.
   - **Zero control:** Collapsed to near-silence (`RMS = 0.0023`, hallucinated `"Thank you."`).
2. **Ears (`Gemma-4-E2B` Conformer + `audio_projector` → 36-layer Transformer):**
   - **Reinjection floor:** Saved `[1, 40, 1536]` projected audio states reproduced downstream transformer hidden states with `0.0` max abs diff and `0.0` RMSE.
   - **Causal swap (`B → A`):** Matched end-to-end `A-vs-B` difference to 8 decimal places (`cos = 0.77120739`, `RMSE = 1.23483789`); zero control shifted RMSE to `1.9285`.
3. **Inkling-2-Mini GPU `EmbedRows` + Zero-Dose Lace Floor:**
   - Native GPU reinjection matched `4/4` exact tokens with `0.0` max abs logit diff and `0.0` KL (vs `0.8494` KL and `1/4` exact tokens on zeroed control).

### Verified Audio Samples (Inline Spectrogram + Playable MP4 & Lossless 24 kHz WAVs)

![Packet C Mouth Gate 0 Causal Swap Spectrogram](samples/mouth_gate0_causal_swap_spectrogram.png)

- [samples/mouth_gate0_causal_swap.mp4](samples/mouth_gate0_causal_swap.mp4) — Playable audio-preview card (`e2e A` → `swap B→A` → `zero control`)
- [samples/mouth_e2e_A.wav](samples/mouth_e2e_A.wav) — `"Order code 407 is confirmed."` (`sha256 54371273...`, identical to native reinjection)
- [samples/mouth_swap_B_into_A.wav](samples/mouth_swap_B_into_A.wav) — `"Sensor gate 928 is closed."` (`sha256 5b7d228b...`, pure continuous hidden-state swap into A)
- [samples/mouth_zero_A.wav](samples/mouth_zero_A.wav) — Zeroed continuous hidden-state control (`RMS 0.0023`, `sha256 068aa066...`)

### 5.2 Stage 3 Mouth (`val_dev`): Prosody Transfers Linearly (`27/32`), Exact Lexemes Do Not (`0/32`) (`att-6c4b19fe`)

On the held-out `val_dev` split (`400` synthesized `24 kHz` clips on RTX PRO 6000 Blackwell, `4.34 GiB` peak VRAM, `1,889.04 s` total, `zero_qwen_text_lookup_at_valdev = true`, `0/160` sealed `confirm_reserve` touched, `key_sha256 cd9a2158...`, report `sha256 b98f45ae...`), closed-form ridge projection from frozen `Inkling-2-Mini` into `Qwen3-TTS-12Hz-1.7B` exposed a sharp functional split. Exact roundtrip reinjection on fit gold clips passed at `0.0` max abs WAV diff (`65,280` and `80,640` samples), eliminating seam or vocoder bugs as confounders. Overall `all_val_dev_gates_passed` is `false` and `confirm_reserve` remains sealed.

#### Exact Counts (`N = 32` Minimal Pairs per Track + `16` Silence Controls)

| Evaluation Track (`val_dev`, `0/160` confirm) | Native `Qwen3-TTS` Ceiling | Inkling Ridge Candidate | Swapped Hidden Control | Zero Hidden Control | Paired Test vs Controls |
|---|---:|---:|---:|---:|---|
| **Style / Prosody Overall (`32` pairs)** | **30/32** (`93.8%`) | **27/32** (`84.4%`) | `5/32` (`15.6%`) | `12/32` (`37.5%`) | vs swapped: **`+22` (`+68.8` pts, 95% CI `[+43.8, +93.8]`, `p = 1.13e-4`)**; vs zero: **`+15` (`+46.9` pts, `p = 1.49e-3`)** |
| — `slow_deliberate` vs `brisk_conversational` | — | **8/8** | `0/8` | `0/8` | — |
| — `calm_neutral` vs `urgent_emphatic` | — | **7/8** | `1/8` | `3/8` | — |
| — `warm_reassuring` vs `terse_clipped` | — | **7/8** | `1/8` | `5/8` | — |
| — `rising_question` vs `falling_declarative` | — | **5/8** | `3/8` | `4/8` | — |
| **Lexical Content Overall (`32` pairs)** | **27/32** (`84.4%`) | **0/32** (`0.0%`) | `0/32` (`0.0%`) | `0/32` (`0.0%`) | `0/8` across every subtype (`digit`, `name`, `negation`, `order`; `p = 1.0`) |
| **EOS / Zero Silence Controls (`16` clips)** | `0/16` FP target | `16/16` Whisper `"Thank you."` | — | — | `8` `eos_only` (`0.56 s`, `RMS 0.000794`) + `8` `zero_cond` (`1.28 s`, `RMS 0.002809`) |

#### Exact Failure Shape

1. **Prosody and cadence survive linear projection; discrete word identities collapse into near-homophone gibberish.** In 4-fold speaker-disjoint cross-validation over the `256` fit clips (`16` speakers), the split-head instruction ridge head achieved `cos = 0.9589` / `R² = 0.8805` (`λ = 30.0`), whereas the lexical text head reached only `cos = 0.7922` / `R² = 0.2387` (`λ = 30,000`; `R² = 0.1713` under the joint head). Downstream blinded `faster-whisper` transcripts on lexical pairs preserve English syllable rhythm while corrupting every critical token:
   - `mouth_val_crit_000` (`digit`: `8520` vs `8250`) → `"Treep Taw 7000 2061 Dradtry"` vs `"Treep 2-1-D-Civito-7 cried side."`
   - `mouth_val_crit_001` (`name`: `Kowalski-1` vs `Patel-1`) → `"Draft Balwar Alavo 1, send RAR."` vs `"Trad alverfua finn send Raymurn."`
   - `mouth_val_crit_002` (`negation`: `still` vs `no longer`) → `"The train and zero traf in slot one."` vs `"The trained Aeon, Braft, and Slot Twin."`
   - `mouth_val_crit_003` (`order`: `Oslo-1` vs `Bergen-1`) → `"Fright Grind going number nine."` vs `"Fright Grind Gurei Benter Nguyen Dambaster Nguyen."`
2. **Unmasked ASR turns near-silent controls into false positives.** Every EOS-only (`RMS = 0.000794`) and zero-conditioning (`RMS = 0.002809`) clip collapsed to near-silence physically, yet `faster-whisper` hallucinated `"Thank you."` on `16/16` clips. Silence-control gates require an explicit pre-ASR RMS energy floor (`RMS >= 0.01`) rather than raw Whisper word counts.

#### Mechanism & Engineering Invariants

Just as Packet B exposed a spatial-frequency boundary on vision (coarse grids carry layout and count linearly but lose crisp glyph pixels), Packet C Stage 3 exposes a **prosody-vs-lexeme boundary on speech synthesis**:

1. **Continuous delivery axes align linearly across frozen models.** Speaking rate (`8/8`), urgency (`7/8`), and warmth (`7/8`) occupy smooth, low-frequency directions in `Qwen3-TTS`'s instruction space (`R² = 0.8805`), allowing a zero-gradient linear map from `Inkling-2-Mini` to retain `90%` of the native directional ceiling (`27/32` vs `30/32`).
2. **Discrete lexical lookup does not tolerate ridge shrinkage.** `Qwen3-TTS`'s `text_projection` seam behaves like a sharp discrete vocabulary lookup (`R² = 0.2387` across mismatched tokenizers). Ridge shrinkage averages nearby subword vectors, leaving cadence intact while destroying exact digits, proper nouns, and negations (`0/32`).
3. **Production speech architecture: discrete tokens for words, Plato's Key for delivery.** A frozen LLM already emits exact discrete text tokens. Routing discrete text tokens to `Qwen3-TTS`'s text input preserves lexical accuracy, while routing the closed-form Plato's Key strictly into the continuous style/instruction seam delivers zero-gradient prosodic and affective control directly from the LLM's internal hidden states.

### 5.3 Second Mouth Seam (`M_explicit_lexical_plus_hidden`, `att-b194ab6a`): Testing the Hybrid Architecture

Under pre-registered authorization (`att-8745190b`) on `mouth_val_dev` (`400` clips on RTX PRO 6000 Blackwell, `1,693.49 s`, `4.34 GiB` peak VRAM, `0/160` confirm reserve touched, `key_sha256 cd9a2158...`, report `sha256 0d799309...`), we supplied identical declared discrete lexical token IDs across all arms while varying only the continuous Inkling-projected `inst_emb` (`matched`, `swapped`, `zero`). The prior direct continuous run (`att-6c4b19fe`) stays preserved as `FAIL`.

| Track (`M_explicit_lexical_plus_hidden`, `val_dev`) | Native Ceiling | Inkling Matched `inst_emb` | Swapped `inst_emb` | Zero `inst_emb` | Paired Exact Test |
|---|---:|---:|---:|---:|---|
| **Style / Prosody Overall (`32` pairs)** | `30/32` (`93.8%`) | **`28/32` (`87.5%`)** | `4/32` (`12.5%`) | `11/32` (`34.4%`) | vs swapped **`p = 1.93e-5`**; vs zero **`p = 1.53e-5`** (`8/8` urgency, `8/8` pace, `6/8` intonation, `6/8` warmth) |
| **Supplied Lexical Preservation (`32` pairs)** | `27/32` (`84.4%`) | **`28/32` (`87.5%`, `103.7%` native)** | `27/32` (`84.4%`) | `0/32` (`0.0%`) | `8/8` digits, `8/8` negation, `8/8` order, `4/8` names; vs zero **`p = 7.45e-9`** |
| **Roundtrip Fit Check (`2` gold clips)** | `0.0` drift | **`0.0` max abs WAV diff** (`2/2`) | — | — | Exact sample-for-sample match (`65,280` & `80,640` samples) |
| **EOS / Zero Silence Controls (`16` clips)** | `0/16` FP | `16/16` Whisper `"Thank you."` | — | — | All `16/16` physically near-silent (`RMS 0.000794–0.002809`) |

Five of six pre-registered `val_dev` gates passed; `all_val_dev_gates_passed` remains `false` (`FAIL 5/6`) solely because un-gated `faster-whisper` hallucinates `"Thank you."` on all `16/16` physically silent controls (`max_rms = 0.002809`). Under a pre-registered prospective scorer amendment (`att-092ddfeb`, `mono float32 16 kHz`, no gain normalization, `RMS >= 0.01 AND duration >= 0.20 s`, applied blind across all arms/clips with gated expected speech counted missed), amended `val_dev` passed `6/6` gates. An independent human blind listening audit over all `32` opaque WAVs (`listening_audit_sheet_LOCKED.md`, `sha256 18a51a06...`, locked prior to unsealing `sealed_audit_key.json`) confirmed **`32/32` (`100%`) binary audibility agreement** with the eligibility gate (`16/16` `eos_zero_control` clips judged `NONE`; `16/16` genuine speech clips judged audible; reported as binary audibility rather than exact word transcription). Original development stays logged as `FAIL 5/6`; one prespecified `M_explicit_lexical_plus_hidden` run on `mouth_confirm_reserve` (`80` items) is now cleared while `ears_confirm_reserve` (`80` items) remains `0/80` sealed.

### 5.4 Visit Lace (`att-d0e96537`), Whisper Ears (`att-773f320b`), & Packet D `YuE2-3B` 48 kHz Music (`att-0fcf223c`)

- **Visit Lace (`att-d0e96537`):** Zero-dose replay matched `64/64` tokens identically; monotonic causal steering scaled `0/32 → 26/32` at `alpha = 3.0` while preserving `32/32` (`100%`) arithmetic reasoning (`p = 1.0`).
- **Whisper-Turbo Continuous Ears (`att-773f320b`):** Closed-form FP64 ridge from `Whisper-large-v3-turbo` continuous `1280`-dim audio features into `Inkling-2-Mini` `2304`-dim `EmbedRows` achieved `1.00000` nonspeech ROC-AUC and `8/8` speaking-pace discrimination.
- **Packet D Native `YuE2-3B` + `YuE2-Vae` (`att-0fcf223c`, `att-c6acea7c`):** Native Gate 0 verified `0.0` max abs diff seam parity and 16-step exact rollout parity, synthesizing a `60.639 s` `48 kHz` stereo anthem (`audio_rms = 0.1679`) in `31.59 s` warm (`0.521×` real-time factor, `8.65 GiB` peak CUDA allocation).

---

## 6. Multi-Model Closed-Form Plato's Key Blends & Cross-LLM Hubs (`#941`, `att-4aa9bbc3`)

Using the exact same frozen `fit` / `val_dev` / sealed `confirm_reserve` split (`160` sealed prompts, `3,413` unseen words, `14/14` frozen SHA-256 pins verified before unsealing, `fit_calls_executed = 0`, `0` GPUs on Quincy CPU), we tested whether combining closed-form representations from heterogeneous donor families (`Inkling-2-Mini` looped MoE + `Qwen3.8-27B-NVFP4` dense) improves conditioning over either model alone, and whether cross-LLM hubs transfer transitively into `MiniMax-H3`.

### 6.1 Sealed Confirm-160 Blend & Transitive Hub Results (`n = 3,413` Words, `16` Disjoint Groups)

| Strategy (`Sealed Confirm-160`, `fit_calls = 0`) | Key Size (FP16) | Word Cos (`n=3413`) | Word `R²` | `rel_l2` | Paired Delta vs Best Single (`Qwen3.8`) |
|---|---:|---:|---:|---:|---|
| Single `Inkling-2-Mini` `concat4` | `90.0 MiB` | `0.92282` | `0.998218` | `0.04070` | baseline |
| Single `Qwen3.8-27B` `concat4` | `200.1 MiB` | `0.95958` | `0.999216` | `0.02699` | best single baseline |
| Single `Qwen3.8-27B` `layer_31` (middle-layer early exit) | `50.0 MiB` | `0.95274` | `0.999090` | `0.02914` | single middle layer retains `99.3%` of `concat4` cos |
| **Mode A: Convex Blend** (`w = 0.1565 Ink + 0.8435 Qwen`) | `290.1 MiB` | **`0.96222`** | **`0.999266`** | `0.02612` | **`+0.00261`** (`CI95 [+0.00198, +0.00349]`, `16/16` groups, `p = 3.052e-5`) |
| **Mode B: `stacked_multikernel_concat4`** (`alpha = [0.1, 0.9]`) | `290.1 MiB` | **`0.96260`** | **`0.999285`** | **`0.02578`** | **`+0.00300`** (`CI95 [+0.00232, +0.00377]`, **`16/16` groups & `8/8` domains**, `typography_signage +0.00641`, `p = 3.052e-5`) |
| **Mode B: `stacked_single_tap_rms` (compact)** | **`72.5 MiB`** | **`0.95860`** | `0.999208` | `0.02714` | **`+0.00584`** vs `Qwen3.8` single tap (`16/16` groups, `p = 3.052e-5`) |
| *Negative finding:* Naive `stacked_raw_concat4` (`val_dev`) | `290.1 MiB` | `0.95622` | — | — | **Regressed `-0.00301`** vs Qwen3.8 due to cross-model hidden-state norm imbalance |
| **Mode C: Hub `Inkling → Qwen3.8(L31)` → `H3`** | `91 MiB` hub | `0.94177` (hub) · **`0.91739`** (→H3) | `0.998053` | — | Transitive conditioning via frozen Qwen3.8 middle layer |
| **Mode C: Hub `Qwen3.8 → Inkling(V41)` → `H3`** | `91 MiB` hub | `0.92945` (hub) · **`0.93407`** (→H3) | **`0.998595`** | — | **Beats Inkling's own direct `concat4` (`0.92282`)** while routing through Inkling's latent space |

### 6.2 Mechanistic Findings & Half-Depth Early-Exit Sealed Confirm-160 Proof (`640` `1024×1024` Images)

1. **Cross-model norm imbalance breaks naive concatenation.** Directly stacking raw hidden states from a looped MoE (`Inkling-2-Mini`) and a dense 27B model (`Qwen3.8-27B`) regressed `-0.00301` on `val_dev` (`0.95622` vs `0.95917`). Applying block-weighted multi-kernel ridge (`alpha = [0.1, 0.9]`, solved via Dual Woodbury in FP64) turned that regression into a uniform `16/16`-group, `8/8`-domain gain on sealed Confirm-160 (`+0.00300`, `p = 3.052e-5`), with the largest domain gain on `typography_signage` (`+0.00641`).
2. **Half-depth single tap (`layer_31 / 64`, `50.02 MiB`) statistically outperforms the `27.14 GB` native `Qwen3-VL-32B` teacher on sealed Confirm-160 (`640` `1024×1024` images).** Evaluating the frozen `Qwen3.8-27B` keys directly on the sealed 160-prompt `MiniMax-H3` image confirmation split (`2× RTX 5090`, `fit_calls_executed = 0`, 4-prompt dev floor `4/4` exact `0.0` RMSE / `99.0 dB`, `per_prompt_outcomes_sha256 142fc216...`):

| Arm (`Qwen3.8-27B` Sealed Confirm-160, `640` `1024×1024` Images) | Key Size (FP16) | RTX 5090 Proj (`32` tok) | SigLIP2 Text-Img (`G=16`) | Margin Retention | Img-vs-Teacher Cos | Final Latent Cos | Paired vs Teacher / `B0` |
|---|---:|---:|---:|---:|---:|---:|---|
| **Native `Qwen3-VL-32B` Teacher Reinject** | `27,141 MB` | — | `0.19838` | `100.00%` | `1.00000` | `1.00000` | `16/16` vs `B0` |
| **`Qwen3.8` `single_tap` (`layer_31 / 64` ONLY)** | **`50.02 MiB`** | **`0.049 ms`** | **`0.20234`** | **`101.55%`** | **`0.90103`** | `0.78411` | **Beats teacher `13/16` groups (`+0.00396`, 95% CI `[+0.00010, +0.00760]`, `p = 0.0213`)**; `16/16` vs `B0` (`p = 3.05e-5`) |
| **`Qwen3.8` `concat4`** *(separate)* | `200.05 MiB` | `0.143 ms` | `0.19992` | `100.60%` | `0.89562` | **`0.78751`** | `9/16` vs teacher (`+0.00154`, `p = 0.804`); `single_tap` beat `concat4` `13/16` on SigLIP2 (`p = 0.0213`) |
| **Matched Shuffled `B0` (`layer_31`)** | `50.02 MiB` | — | `-0.05670` | `0.00%` | `0.48249` | `0.34597` | — |

   *Scope boundary:* This proves functionally across `640` unseen `1024×1024` images that the back half of a 64-layer dense donor LLM (`layers 32–63`, final norm, and LM head) is unnecessary for image diffusion conditioning, and that `50.02 MiB` `layer_31` outperforms both the `27.14 GB` teacher (`p = 0.0213`) and `200.05 MiB` `concat4` (`p = 0.0213`) on SigLIP2 prompt adherence. End-to-end wall-clock early-exit prefill speedups and per-layer sweeps on `Inkling-2-Mini` / `FL2VA` / `Mouth` remain in `#944` (not yet measured).

---

## 7. Cross-Silicon Validation: 2017 Volta V100 → Blackwell & 3-Device Mobile Mesh

### 7.1 Wave 3 Real End-to-End Image + `5.875 s` Video+Audio (Zero 32B Encoder)

Every local generation below ran real `1024×1024` `MiniMax-H3` image generation and real `5.875 s` (`141`-frame `768×448`) `FL2VA` video + `32 kHz` stereo audio generation conditioned solely by the frozen `90.0 MiB` Inkling `concat4` Plato's Key (`cos = 0.999996` vs FP64 CPU, zero `Qwen3-VL-32B` loaded):

| GPU (Architecture, VRAM) | Plato's Key (`32` tok) | Real `1024×1024` Image (DiT Format, Peak) | Real `5.875 s` `141f` `FL2VA` Video + `32 kHz` Audio (Peak, RMS, Motion L1) | Receipt |
|---|---:|---|---|---|
| **2017 Tesla V100** (`sm_70`, `31.73 GiB`) | `1.566 ms` | **`29.31 s`** (`int8_convrot`, `22.32 GiB`) | **`796.01 s`** (`27.02 GiB`, `audio_rms = 0.012029`, `motion_l1 = 8.147`) | `att-717e4057` |
| **RTX 3090** (`sm_86`, `23.56 GiB`) | `0.121 ms` | **`71.52 s`** (`int8_convrot`, `20.77 GiB`) | **`178.62 s`** (`22.09 GiB`, `audio_rms = 0.012009`, `motion_l1 = 8.153`) | `att-82078ba5` |
| **RTX PRO 6000 Blackwell** (`sm_120`, `95 GiB`) | `0.022 ms` | **`27.68 s`** (`nvfp4`, `12.95 GiB`) | **`99.61 s`** (`22.97 GiB`, `audio_rms = 0.011936`, `motion_l1 = 8.157`) | `att-af0ed07a` |

![Wave 3 Real Image & 5.875s Video+Audio Across Generations](figures/omni912_wave3_real_image_and_5s_av_benchmark.png)

### 7.2 Stage 2 24-GPU Fleet Sweep (All 6 CUDA Generations, `att-41cf9f0c`)

Across `4` local cards (`att-98b7c6d6`) plus `20` bounded Vast instances (`att-74557f1b` Wave 1 + `att-41cf9f0c` Wave 2; `all_teardowns_zero_confirmed = true`), **24 distinct real NVIDIA GPUs spanning all 6 CUDA generations (`Pascal → Volta → Turing → Ampere → Ada → Blackwell`)** verified Plato's Key projection parity, native `SigLIP2` Sight, native `Qwen3-TTS` Voice, and H3-width GEMM throughput (explicitly scoped to component parity + GEMM microbench; full Wave 3 real Image+FL2VA Vast sweep across Ampere/Ada/Blackwell is running separately).

![24-GPU Multi-Card Omni-Modal Latencies](figures/omni912_multicard_omnimodal_latencies.png)

### 7.3 On-Device Android Voltron 3-Device Neural Mesh & Adreno 840 Cooperative Matrix (`#914`)

- **Live 3-Device Real Neural Pipeline Over Wi-Fi LAN (`att-111b1358`, zero cloud, zero server GPUs):** `Galaxy S26` Brain (`Inkling-2-Mini EXL3-9g` on Adreno 840, `9.81 tok/s`; Track 2 Iter 8 peak `10.463 tok/s` Pass 2, `att-81f2ecd2`) → `Galaxy Tab S8+` Neural Voice (`Kokoro-82M INT8 ONNX`, `8.03 s` `24 kHz` speech in `5.74 s` warm = `1.40×` faster than real-time) + `Galaxy S24 Ultra` Neural Artist (`1-step SDXS-512 Q8_0 GGUF` via `stable-diffusion.cpp`, `37.88 s` warm co-hosted alongside `Inkling-2-Mini` in `4.40 GB` total PSS, zero LMK kills; currently text-prompted while verifying the `64 KiB` packet SHA-256).
- **Mobile DiT Silicon Measurement (`att-10b33ab9`) & VPKE v1 Interop (`att-320e7c1e`):** Live Vulkan benchmarks of `Tongyi-MAI/Z-Image-Turbo` (`6.15B` S3-DiT) linear shapes confirmed hardware `VK_KHR_cooperative_matrix` (`64×64×16 FP16`) on `Galaxy S26` Adreno 840 at `2.50–2.68` FP16 TFLOPS on `K=3840` projections and `1.04` TFLOPS on `W2` (`49.21 s` isolated 30-block × 8-forward `512px` linear subtotal, vs `218.32 s` SIMT on Adreno 750 `S24 Ultra` without coopmat). Python ↔ Kotlin VPKE v1 packet packing verified byte-identical (`328,233 B`, `seq_len=64, dim=2560`, `30/30` JVM tests) to receive the `#943` `Z-Image-Turbo` Plato's Key directly on device.

---

## 8. System Footprint & Consumer-GPU Targets

All sizes below use **decimal GB (`10^9` bytes)** consistently and separate **measured files / peak VRAM** from **unmeasured projection targets**. Because `MiniMax-H3`'s full **Audio+Video DiT** (`minimax_h3_fl2va_pruned_int8_convrot.safetensors`, `21.0 GB` decimal / `20 GiB` on disk, plus `3.2 GB` video VAE and `0.61 GB` audio VAE) conditions on the **exact same `Qwen3-VL-32B` Layer-50 `5120`-dim space** as `MiniMax-H3` Image, the **exact same `94.4 MB` (`90.0 MiB`) Packet A Plato's Key** targets native video + synchronized audio generation at **zero extra bridge bytes** (functional video temporal-coherence gate `PENDING`). An already-quantized **`12.54 GB` (`11.68 GiB`, `12,537,569,802` bytes) NVFP4** H3 Image DiT (`minimax_h3_image_pruned_nvfp4.safetensors`) is also on disk.

### 8.1 Measured Baseline vs Packet A Replacement (Decimal GB)

| Component | Measured Disk (GB) | Measured Peak VRAM (GB) | Status |
|---|---:|---:|---|
| Eliminated Teacher (`Qwen3-VL-32B` L50 INT8) | **27.14** | **31.47** | Replaced by `0.094 GB` Packet A `concat4` key (`288×` smaller) |
| `Inkling-2-Mini` NVFP4 Donor Trunk | **19.53** | **21.82** | Measured isolated (`Packet A` report) |
| `MiniMax-H3` Image DiT INT8 + FP16 VAE | **22.15** (`20.98 + 1.17`) | **25.55** | Measured isolated (`Packet A` report) |
| Packet A `concat4` / `single_tap` Bridges | **0.094** / **0.024** | — | Fitted FP16 `.safetensors` files (`att-f55d4648`) |
| Sight (`GOT-OCR2` stripped vision tower) | **0.191** | **1.56** *(Stage 1 extract)* | `620.5M` decoder stripped (`att-5a611ab6`) |
| Unstripped Ears (`Gemma-4-E2B` INT8) | **5.20** | — | Full checkpoint on disk (Conformer not yet stripped) |
| Unquantized Mouth (`Qwen3-TTS-1.7B` BF16 + codec) | **~5.31** (`4.62 + 0.68`) | — | Full BF16 weights on disk (not yet quantized) |

### 8.2 Projected Stripped + Quantized Deployment Targets (Not Yet Measured on Silicon)

*(Bridge sizes for B2 Sight `~0.016 GB`, C Ears `0.007 GB`, and C Mouth `0.038 GB` are FP16 shape arithmetic until fitted; Packet A `0.094 GB` is a measured file. Disk weights always understate live VRAM due to KV cache and activations.)*

| Component | Target Weights (Decimal GB) | Precision / Engineering Rule |
|---|---:|---|
| `Inkling-2-Mini` Elastic Trunk (`atlas-6g` / `atlas-8g` / `atlas-10.5g`) | **6.0–10.5** *(<=8.5 GB required for 12 GB residency)* | Frozen trunk (`0.0%` forgetting) |
| All 4 Plato's Key Bridges (`A` measured + `B2/C` shape arithmetic) | **0.16** (`0.094 + 0.016 + 0.007 + 0.038`) | Keep FP16 |
| Sight (`GOT-OCR2` stripped tower, measured on disk) | **0.19** | Already stripped |
| Ears (`Gemma-4-E2B` stripped Conformer, target) | **0.40–0.60** | Strip text LLM, keep Conformer |
| Mouth (`Qwen3-TTS-1.7B` quantized talker + FP16 codec, target) | **1.50–1.80** | Quantize talker; keep `0.68 GB` codec FP16 |
| **Voice + Sight + Text Weight Sum (exact row sum)** | **8.25–13.25** *(**8.25–11.25** with `<=8.5 GB` trunk)* | **12 GB RTX 3060 residency requires `<=8.5 GB` trunk** (leaving KV/activation headroom) |
| Diffusion Image (`MiniMax-H3` NVFP4 DiT `12.54 GB` on disk + `1.17 GB` VAE) | **13.71** *(further packing target `~10–11 GB` not yet built)* | Keep `1.17 GB` VAE FP16 |
| **Full 4-Modality Weight Sum (+ On-Disk NVFP4 H3 Image)** | **21.96–26.96** | Single 24–32 GB card resident (`<=8.5 GB` trunk); 12 GB 3060 requires block-level DiT streaming or further packing to `<=10–11 GB` |

![Measured Footprint vs Target Ladder](figures/fig3_omni_footprint_ladder.png)

---

## 9. Active & Remaining Work

- **Showcase 1 v4 (`60.64 s` Anthem) & Showcase 2 v4 (Real Blender 5.2.5 Cycles 3D Room + `Ref2VA`):** Active on Devbox RTX PRO 6000 (`Showcase 2 v3` was logged honestly as visual-QA `FAIL` after human inspection caught 2D collage geometry drift and a hard jump-cut at `yaw=360°`, motivating v4's real 3D Blender anchor views and fail-closed jump-cut + SigLIP2 identity gates).
- **Packet C Mouth Scorer Amendment (`val_dev`) & `#942` Public Expressive TTS / `YuE2` Singing Shoot-Out:** Frozen v2 `RMS >= 0.01` CPU audit running on Quincy (`confirm_reserve` `0/160` sealed); `#942` 5-engine TTS + `YuE2` melancholic singing shoot-out live on Vast L40S (`Inkling → IndexTTS-2` 8-float OOF `cos = 0.9144`, `Qwen3.8 → IndexTTS-2` `cos = 0.9362`).
- **3-Way & 4-Way Multi-Model Atlas Blends (`#941`):** `DeepSeek-V4.1-Flash` and `GLM-5.3-Flash` taps completing to join `Inkling-2-Mini + Qwen3.8-27B`.
- **Universal Nov-2026 Base/Turbo Image Plato's Key Sweep (`#943`, BYO-Quant, `V100 → Blackwell`):** Phase 0 frozen (`att-74ce97b4`); Wave A (`Qwen-Image-2.1`, `Krea-2`, `FLUX.3-Image-Klein`, `Z-Image-Turbo`, `FLUX.2-klein-4B`) + Wave B (`FLUX.2-klein-9B/dev`, `P-Image-OSS`, `MOSS-Image`, `FLUX.1-schnell`, `SDXL-Turbo`) queued on Quincy V100 / Blackwell.
- **Ultra-Sparse Multimedia Mode Probe (`#944`):** Measuring Brain early-exit depth + MoE media-expert sparsity + Plato's Key SVD rank and Desktop GPU / S26 DiT block-skip, `2:4` / SwiGLU FFN sparsity, and `FL2VA` spatial-temporal patch sparsity.
- **Wave 3 Vast Sweep:** Completing real Image + `5.875 s` `FL2VA` Video+Audio across 11 Ampere/Ada/Blackwell Vast SKUs.
- **Process Trace & Centaur History:** Maintained in [PROCESS_LOG.md](PROCESS_LOG.md).

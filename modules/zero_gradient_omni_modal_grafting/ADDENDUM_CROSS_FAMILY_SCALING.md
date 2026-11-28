# Addendum: Zero-Gradient Cross-Family Manifold Scaling — Hypothesis & Adversarial Falsification Plan

**Status:** Working Addendum & Preregistered Falsification Protocol (`confirm_reserve` untouched for `N=4, 5`; `0/160`)  
**Parent Paper:** *Omni by Atlas: Zero-Gradient Omni-Modal Grafting via Closed-Form Plato's Keys* (`v0.9.3`)  
**Author:** William Joseph Tatum (Monkey)  
**Date:** 2026-11-27

---

## 1. Observed Effect (Measured Through `N = 5` Independent LLM Families)

When frozen hidden states from $N$ distinct open-weight LLM families are combined through exact FP64 multi-kernel dual Ridge regression ($K = \sum_{i=1}^{N} \alpha_i K_i$, $\alpha_i \ge 0$, $\sum_i \alpha_i = 1$, zero gradient steps, zero trunk weight edits) to predict MiniMax-H3 text-conditioning embeddings (`concat_all4` taps per trunk), held-out `val_dev` accuracy improves monotonically at every step from `N = 1` to `N = 5` (`160` held-out prompts, `3,446` words; `confirm_reserve = 0/160` untouched for `N = 4, 5`):

| $N$ | Added Independent Family | Single-Model `val_dev` Cos (`concat_all4`) | `stacked_multikernel` `val_dev` Cos | `stacked_multikernel` Sealed `Confirm-160` Cos | Notes / Verified Evidence |
|---:|---|---:|---:|---:|---|
| 1 | `Qwen3.8-27B-NVFP4` | `0.95917` | `0.95917` | *(single baseline)* | `val_dev` `r2=0.999218`, `rel_l2=0.02697` |
| 2 | `+ Inkling-2-Mini` | `0.91455` | `0.96159` | **`0.96260`** (`rel_l2=0.02578`) | `out_blend2_inkling_qwen38` (sealed Confirm-160 verified) |
| 3 | `+ DeepSeek-V4.1-Flash` | `0.84470` | `0.96247` | **`0.96320`** (`rel_l2=0.02563`) | `out_blend3_qwen_ink_ds41` (sealed Confirm-160 verified) |
| 4 | `+ GLM-5.3-Flash` | `0.93426` | `0.96419` | *held (`0/160`)* | `out_blend4_qwen_ink_ds41_glm53` (`val_dev` only) |
| 5 | `+ Ling-3.2-air` | — | **`0.96854`** | *held (`0/160`)* | `job-1127130833-ae74c9`: `r2=0.999423`, `rel_l2=0.02316` (`-14.1%` rel_l2 vs Qwen3.8; 16-group paired cosine `p = 3.052e-5` vs Qwen3.8) |

Two structural details stand out in the measured data:

1. **Every independent family receives substantial positive weight at `N = 5`:** $\alpha = [0.475\ \text{Qwen3.8},\ 0.216\ \text{Ling-3.2-air},\ 0.139\ \text{GLM-5.3-Flash},\ 0.118\ \text{Inkling-2-Mini},\ 0.051\ \text{DeepSeek-V4.1}]$. Even the weakest single trunk (`DeepSeek-V4.1`, `0.8447` standalone) retains `5.1%` optimal kernel weight, and the 3-way `stacked_multikernel` blend beats the best single model (`Qwen3.8`) across all 8 prompt domains (`+0.0025` to `+0.0039` cosine vs best single model; `out_nplus1_real_3way`).
2. **Naive concatenation fails; per-family kernel weighting is mandatory.** In `out_nplus1_real_3way`, naive `stacked_raw` regresses at `N = 2` (`0.95919 -> 0.95621`), `stacked_rms_norm` regresses at `N = 3` (`0.96033 -> 0.95739`), and equal-weight `convex_equal` collapses (`0.95919 -> 0.94846`). Only multi-kernel Ridge (`stacked_multikernel`) and MSE-optimal convex combination (`convex_mse_opt`) remain strictly monotonic.

---

## 2. Candidate Hypothesis (What Must Survive Falsification)

**Zero-Gradient Cross-Family Manifold Scaling:** Distinct frozen LLM pretraining lineages span partially orthogonal linear subspaces of a shared multimodal target manifold. Under exact per-family multi-kernel regularization (preventing noisier trunks from flooding the Gram matrix), held-out target reconstruction error $E(N) = \text{rel\_l2}(N)$ decreases monotonically with the number of independent model families $N$, **above and beyond** matching feature dimensionality, same-family multi-layer depth, or variance-reduction bagging — and translates to measurable out-of-domain and end-to-end generative fidelity gains.

We explicitly do **not** claim a proven scaling law from 5 `val_dev` points. The battery below is designed to spend compute trying to **disprove** this hypothesis by testing every mundane confound that could mimic the `N = 1 -> 5` curve.

---

## 3. Adversarial Falsification Battery (6 Pre-Committed Kill Tests)

Each test targets one specific rival explanation. All model selection and hyperparameter fitting occur strictly on `fit` + `val_dev`; `confirm_reserve` (`160` prompts, `16` source groups) remains `0/160` **for `N = 4, 5`** (earlier sealed post-freeze `confirm_reserve` evaluations exist for `N = 2, 3` as listed in Section 1), and `#941`/`#957` `confirm_reserve` access for `N = 4, 5` is coordinated under **one single later decision** once frozen and cleared by `astartes-maxima`.

| Kill ID | Rival Explanation Targeted | Concrete Adversarial Experiment | Pre-Committed Falsification Criterion (Kills / Bounds Claim) |
|---|---|---|---|
| **F1** | **Feature-Dimension Bloat** (*"More dimensions $D$, not more families"*) | (a) Expand single-family `Qwen3.8-27B` from 4 taps (`20,480` dim) to **8, 12, and 16 taps** (`40,960`–`81,920` dim, matching/exceeding 5-way total width). (b) **Feasible common-rank comparison (labeled separately, strictly fold-local centering + SVD/PCA):** prespecified primary rank **`k_primary = 4096`** (verified strictly $\le$ measured numerical rank of both arms in every training fold); `k ∈ {1024, 2048, 8192}` labeled **sensitivity analyses**; full-fit maximum ($\le 13,871$, verified numerically) reported separately. | Prespecify practical gain margin **`delta_min = +0.0020` cosine** (or `>= 2.0%` relative `rel_l2` reduction) over 16 paired source groups (95% paired CI): (1) **Falsified / Equivalence** if the upper 95% CI bound on (5-way minus Qwen3.8 at `k_primary`) is `< delta_min`, OR if single-family `Qwen3.8` at matched/higher $D$ matches or beats the 5-way blend within `delta_min`; (2) **Inconclusive** (fails advancement without establishing equivalence) if two-sided $p \ge 0.05$ while the 95% CI straddles `delta_min`. |
| **F2** | **Lucky Ordering / Best-First Cherry-Picking** | Enumerate **all $\sum_{k=1}^{5}\binom{5}{k} = 31$ non-empty subsets** on `val_dev` (all $120$ addition orders), plus (a) strict leave-one-out drop from `N = 5` for every family, and (b) the 4-family ladder **excluding `Qwen3.8`** (`N = 1..4`). | Hypothesis **FALSIFIED / BOUNDED** if subset-averaged $E(k)$ is not strictly decreasing across $k=1..5$, if the 4-family no-Qwen ladder fails to decrease monotonically, or if leave-one-out shows only 2 families carry the entire gain. |
| **F3** | **Bagging / Shrinkage / Noise-Averaging Artifact** | Compare real 5-way cross-family blend against three same-width controls solved by the **identical** multi-kernel Ridge pipeline: (a) `B0_shuffled` (prompt-shuffled taps for families 2..5), (b) `B_clone` (4 variance-matched perturbed/quantized copies of `Qwen3.8`), and (c) **same-family multi-checkpoint pair** (e.g., two Qwen checkpoints vs two distinct families). | Hypothesis **REDUCED TO BAGGING** if `B_clone` or same-family multi-checkpoint blends capture $\ge 70\%$ of the `rel_l2` reduction achieved by distinct pretraining families, or if `B0_shuffled` receives non-trivial $\alpha_i > 0.02$ or beats `N = 1`. |
| **F4** | **Single-Teacher / MiniMax-H3 Artifact** | Run the identical `N = 1 -> 5` multi-kernel Ridge ladder against **at least 2 independent non-H3 targets** using existing frozen teacher taps: **`FLUX.2-dev` (Mistral-24B)** and **`Qwen3-TTS` (Mouth) / `YuE2-3B`**. | Hypothesis **TARGET-SPECIFIC (NOT GENERAL)** if the monotonic `N = 1 -> 5` `rel_l2` reduction fails to replicate on independent non-H3 target manifolds. |
| **F5** | **In-Distribution Overfit & Perceptual Saturation** (*"Embedding gain dies before pixels"*) | (a) **Leave-One-Domain-Out (LODO) OOD CV** across all 8 prompt domains (fit on 7 domains, test on the unseen 8th). (b) Single frozen pass on sealed **`confirm_reserve` (160 prompts)** after `astartes-maxima` GO. (c) **End-to-end rendered H3 image/AV evaluation** (Qwen3.8 pure vs 5-way Atlas Blend on matched seeds/prompts, scored by independent SigLIP2/CLIP/human blind check). | Claim **BOUNDED TO EMBEDDINGS** if LODO OOD transfer collapses OR if `+0.00934` embedding cosine yields $\le 0$ measurable improvement in downstream rendered H3 visual/AV fidelity. |
| **F6** | **Fast Finite-Basis Saturation ($N \ge 6\text{–}8$)** | Add 2–3 genuinely distinct additional open-weight families (e.g., **Gemma-3 / Llama-3 / Mistral / Phi-4**) to extend $N = 1..7/8$. Fit both power-law $E(N) = E_\infty + c N^{-\alpha}$ and fast-saturating exponential $E(N) = E_\infty + c e^{-\beta N}$ via leave-future-$N$-out prediction (fit on $N=1..4$, predict $N=5..8$). | Claim **NOT AN UNBOUNDED SCALING LAW** if error plateaus flat by $N=4\text{–}5$ ($\Delta\text{rel\_l2}_{5\to 7} < 1\%$ relative or non-significant) — in which case we report exact **finite-subspace saturation dimension $N^*$**. |

---

## 4. Compute & Execution Plan

- **Phase 0 (CPU-only on Quincy, `$0.00` Vast — conditional GO from `astartes-maxima` after `job-1127130833-ae74c9` finishes):**
  - Run **F1(b)** (strictly fold-local centering + SVD/PCA; prespecified primary rank **`k_primary = 4096`** verified $\le$ measured numerical rank of both arms in every training fold; `k ∈ {1024, 2048, 8192}` labeled sensitivity analyses; full-fit max $\le 13,871$ reported separately; 16-group 95% paired CI against `delta_min = +0.0020` cosine so `p >= 0.05` is never conflated with equivalence), **F2** (all 31 subsets + leave-one-out + no-Qwen ladder), **F3(a,b)** (`B0_shuffled` + `B_clone`), and **F5(a)** (8-domain LODO OOD CV) on the already-cached 5-family `fit/val_dev` taps. Zero new model downloads, zero GPU cost, `confirm_reserve = 0/160` for `N = 4, 5`.
- **Phase 1 (Local GPU / Zero-Cost Colab G4 PRO 6000 — after `astartes-maxima` GO):**
  - **F1(a):** Extract 16-layer `Qwen3.8-27B` taps on `fit/val_dev` (same 800 prompts) and test 4 vs 8 vs 12 vs 16 layers against the 5-way blend.
  - **F4:** Run `N = 1 -> 5` multi-kernel scaling on frozen `FLUX.2-dev` (Mistral-24B) and `Qwen3-TTS` / `YuE2` targets.
  - **F5(b,c):** Single frozen post-seal `confirm_reserve` (160 prompts) evaluation + matched-seed H3 visual/AV render showdown (`Qwen3.8` pure vs 5-way blend).
- **Phase 2 (Extension to $N = 6..8$ Families — Conditional on Surviving F1–F5):**
  - Only if F1–F5 fail to kill the hypothesis, extract 2–3 additional distinct open-weight families (e.g., Llama-3, Gemma-3, Mistral) on the identical 800 prompts and run **F6** curve fitting ($N = 1..8$).

---

## 5. Publication Commitment

Every kill test outcome — whether the hypothesis survives or gets disproved by F1 (dimension bloat), F3 (bagging), F4 (H3-only artifact), F5 (pixel saturation), or F6 (early plateau) — will be published verbatim in this addendum with exact SHA-256 artifact hashes and paired p-values.

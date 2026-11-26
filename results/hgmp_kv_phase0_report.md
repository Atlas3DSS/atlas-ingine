# Phase 0 Empirical Report: Loop-Pass Homology Falsification, Hopper Q4, and Cold-Sketch Rehydration in Inkling-2-Mini KV Compression

**Study ID:** Switchboard `#880` (Publication `#892`, Website Handoff `#893`)
**Date:** 2026-11-26
**Model:** `Inkling-2-Mini` (`atlas-ingine`, 64 attention layers, 40 full-attention visits across 8 KV heads, `D = 128`, 24 sliding-window layers kept uncompressed)
**Worktree Branch:** `feat/hgmp-kv-880`
**Verdict on Primary Hypothesis (H1/H2):** **FALSIFIED** on both held-out evaluations (`phase0a_prelim_32k_64k` FP16 and `phase0b_confirm_32k_256k` BF16).

---

## Abstract

We tested whether loop-pass representational stability inside `Inkling-2-Mini` can rank KV-cache tokens for aggressive mixed-precision compression better than attention-history eviction (`H2O/MiKV`). On 18 development prompts (16K to 32K), we froze the homology direction as stability (`B1_HGMP_Stab`) before inspecting any held-out pack. Across 23 held-out FP16 prompts (32K to 64K) and 36 confirmatory BF16 prompts (32K to 256K), loop-pass homology falsified both primary hypotheses. At 3.25 bytes per element (`b/e`), `B1_HGMP_Stab` trailed `B4_H2O_MiKV` by 9.92 percentage points in FP16 (`n = 20` primary prompts) and by 5.34 percentage points in BF16 (`n = 32` primary prompts; 95% CI `[-14.75, +1.48]`), failing the pre-registered `+5.0` point win bar. Strict all-4 binary scoring on multi-value needles widened the deficit to 7.69 points (`[-18.75, +1.08]`).

Two positive mechanisms survived Phase 0. First, Hadamard-rotated and centered 4-bit quantization (`A2_Q4_Hopper`, 4.5010 b/e including sidecar bytes) recovered `+7.38` points over standard 4-bit quantization (`A1_Q4`, 4.5000 b/e) across all 36 confirmatory prompts (`[-0.88, +17.60]`) and lowered teacher-forced KL from `0.005471` to `0.004871`. Follow-on dev-only calibration (`Gate 1`, `n = 18` BF16) showed that strictly prefix-causal `W = 512` centering with 4 full-precision prefix sink tokens (`M1_CausalW512_s4`, zero external calibration) cuts primary teacher-forced KL by 18.6% relative to `A1_Q4` (`0.002851` vs `0.003502`). Second, query-conditioned cold-page rehydration (`B7_Rehydrated_Top25pct`) lifted task accuracy by `+54.16` points at 32K to 256K (`0.0349` to `0.5765`, page AUROC `0.7809`, Recall@25% `0.6556`) and reached page AUROC `0.9961` with `1.00` Recall@4 across three separate 1M-token needle prompts (rank 2 on every depth), under reference-conditioned question queries and an expanded active working set (`2.60` to `5.77 b/e`). We drop loop-pass homology tiering, ship causal Hopper Q4 with Verified-Ledger pinning in Phase 1, and defer live unconditioned block-sparse Top-K decode to Phase 2.

---

## 1. Question

Can per-token loop-pass homology (representation stability across recurrent passes) rank full-attention KV entries for mixed-precision tiering more accurately than attention-history baselines (`H2O/MiKV`) at equal byte budgets, and which alternative compression or retrieval mechanisms hold up from 32K to 1M tokens?

---

## 2. Rival Hypotheses, Baselines, and Pre-Registered Rules

Every evaluation arm restores from an identical uncompressed `ActT` KV snapshot (`C0_ActT`, FP16 on V100 or BF16 on Blackwell) and checks restore identity (`C0_ActT` self-KL must equal `0.0` at all 32 teacher-forced positions on every prompt).

### 2.1 Hypotheses Tested

1. **H1/H2 (Loop-Pass Homology Tiering):** Using loop-pass stability (`B1_HGMP_Stab`) to allocate Hot (`ActT`), Warm (`Q4_0`), and Cold (`Q2_g64`) tokens at `3.25 b/e` beats `B4_H2O_MiKV` by at least `+5.0` points on primary non-ledger prompts and exceeds `0.75` needle AUROC.
2. **Regime A (4.5 b/e Uniform vs Rotated Q4):** Hadamard-rotated and mean-centered 4-bit keys (`A2_Q4_Hopper`) reduce quantization loss relative to stock `Q4_0` (`A1_Q4`). Because Phase 0 stored per-visit key means and header metadata outside the `4.5000 b/e` ceiling (`+118.8 KB` to `+380.4 KB`), `A2_Q4_Hopper` is labeled an over-budget comparator in Phase 0.
3. **Regime C (Verified-Ledger Pinning):** Explicitly pinning Verified-Ledger spans at `ActT` inside a `3.25 b/e` budget (`C_Ledger_Pinned`) preserves exact recall under aggressive compression.
4. **H4 (Cold-Sketch Page Retrieval and Rehydration):** Scoring cold 4096-token pages using a `Q1K/Q2V` cold sketch (`2.60 b/e`) and rehydrating the top 25% of cold pages into `ActT` (`B7_Rehydrated_Top25pct`) recovers long-range retrieval. Pre-registered deployment bar: Recall@25% `>= 0.90`.

### 2.2 Baselines and Byte Budgets

| Regime | Arm | Precision / Policy | Target b/e | Byte Accounting Note |
| :--- | :--- | :--- | ---: | :--- |
| Reference | `C0_ActT` | Full uncompressed `ActT` (FP16 or BF16) | `16.00` | Zero-KL restore baseline |
| Reference | `C1_FP8` | Uniform FP8 E4M3 | `8.00` | Exact target budget |
| Regime A | `A1_Q4` | Uniform stock `Q4_0` (`g = 32`) | `4.50` | Exact target budget |
| Regime A | `A2_Q4_Hopper` | WHT-128 rotated + centered `Q4_0` | `4.50*` | Over-budget comparator (`4.5002` to `4.5016 b/e`) |
| Regime A | `A3_HGMP_Stab` / `A6_H2O_MiKV` / `A8_Oracle_MiKV` | Tiered `4.50 b/e` comparators | `4.50` | Within target (`delta <= 0 B`) |
| Regime B | `B0_Q2_g64` | Uniform 2-bit (`g = 64`) | `2.50` | Exact payload + scale headers |
| Regime B | `B1_HGMP_Stab` | Loop-pass stability tiering (frozen primary) | `3.25` | Strictly within target (`delta <= 0 B`) |
| Regime B | `B2_HGMP_Drift` / `B3_HGMP_KDev` | Drift and key-deviation homology ablations | `3.25` | Strictly within target (`delta <= 0 B`) |
| Regime B | `B4_H2O_MiKV` | Cumulative attention-history tiering | `3.25` | Strictly within target (`delta <= 0 B`) |
| Regime B | `B5_Random_s{0,1,2}` | Three-seed uniform random tiering | `3.25` | Strictly within target (`delta <= 0 B`) |
| Regime B | `B6_Oracle_MiKV` | Future-attention oracle upper bound | `3.25` | Strictly within target (`delta <= 0 B`) |
| Regime B/H4 | `B7_ColdSketch_Q1K_Q2V` | Boundary-protected `Q1K/Q2V` cold sketch | `2.60` | Strictly within target (`delta <= 0 B`) |
| Regime B/H4 | `B7_Rehydrated_Top25pct` | `B7` + top 25% cold pages rehydrated to `ActT` | `5.68` to `5.77` | Expands active working set (`+239 MiB` to `+6.40 GiB`) |
| Regime C | `C_Ledger_Pinned` | Verified-Ledger spans pinned at `ActT` | `3.25` | Strictly within target (`delta <= 0 B`) |

### 2.3 Benchmark Packs

| Pack | Items | Suites and Lengths | SHA-256 |
| :--- | ---: | :--- | :--- |
| `phase0a_dev_16k_32k.json` | 18 | 6 MRCR-8N, 6 MK-NIAH, 2 MV-NIAH, 2 VT, 2 Ledger (16K to 32K) | `c96d33de814daac2c69d8f3f4db0179680809897908154a259fbc9f15c22341d` |
| `phase0a_prelim_32k_64k.json` | 23 | 8 MRCR-8N, 8 MK-NIAH, 2 MV-NIAH, 2 VT, 3 Ledger (32K to 64K) | `5e08ee752797329e1769b6e39a11e1ff719a005c6dde8667bf8dc6811aea576b` |
| `phase0b_confirm_32k_256k.json` | 36 | 16 MRCR-8N, 8 MK-NIAH, 4 MV-NIAH, 4 VT, 4 Ledger (32K to 256K) | `bf71437edf131b365bf77a9f2051e1e44363bad1fcd480949302451e0dfd6451` |
| `phase0b_needle_1m.json` | 3 | 3 MK-NIAH needles at depths 15%, 50%, 85% (`~1.05M` to `1.08M` tok) | `5503f0eb30bdbca0cee65c91c07ec3960ee0e9bb7f3424995c5bf9143c13ed88` |

---

## 3. Measured Findings

All four packs passed restore identity with `C0_ActT` self-KL equal to `0.0` at every evaluated position (576 dev, 736 prelim, 1,152 confirm, 96 1M) and `validation_errors = []`.

### 3.1 Pre-Registered Direction Freeze on Dev Pack (FP16, V100)

On `phase0a_dev_16k_32k` (`n = 16` primary non-ledger prompts, `385,016` input tokens, V100-PCIE-32GB FP16, `att-048619a3`):

- Stability (`B1_HGMP_Stab`): needle AUROC `0.6744 [0.5810, 0.7599]`, primary score `0.5735` (all-18 `0.6209`), paired vs Random `+0.1434 [+0.0177, +0.2862]`, paired vs `B4_H2O_MiKV` `-0.0299 [-0.2006, +0.1257]`.
- Drift (`B2_HGMP_Drift`): needle AUROC `0.3256 [0.2401, 0.4190]`, primary score `0.2783` (all-18 `0.3584`), paired vs Random `-0.1518 [-0.3147, -0.0123]`.

We froze `hgmp_direction = "stability"` (`phase0_frozen_direction.json`, SHA-256 `a6f83b7b`) prior to reading any held-out output.

### 3.2 Held-Out H1/H2 Evaluation (FP16 Preliminary and BF16 Confirmatory)

| Metric | Prelim FP16 (`phase0a_prelim_32k_64k`) | Confirm BF16 (`phase0b_confirm_32k_256k`) | Pre-Registered Bar |
| :--- | :--- | :--- | :--- |
| Primary non-ledger count | `n = 20` (all `23`) | `n = 32` (all `36`) | Complete pack |
| `B1_HGMP_Stab` primary score (all) | `0.4711` (`0.5401`) | `0.4228` (`0.4869`) | Higher is better |
| `B4_H2O_MiKV` primary score (all) | `0.5704` (`0.6264`) | `0.4762` (`0.5344`) | Comparator |
| `B5_Random` 3-seed primary mean (all) | `0.5725` (`0.5993`) | `0.3019` (`0.3795`) | Comparator |
| Paired `B1 - B4_H2O_MiKV` (95% CI) | `-0.0992 [-0.2381, +0.0104]` | `-0.0534 [-0.1475, +0.0148]` | `>= +0.0500` |
| Paired `B1 - B5_Random` (95% CI) | `-0.1014 [-0.2357, +0.0132]` | `+0.1209 [+0.0099, +0.2492]` | Beat random |
| Needle token AUROC (95% CI) | `0.6744 [0.5818, 0.7593]` | `0.6810 [0.6173, 0.7424]` | `>= 0.7500` |
| Verdict on H1/H2 | **FALSIFIED** | **FALSIFIED** | Pass both bars |

**Strict All-4 Binary MV-NIAH Sensitivity (`att-82a19df1`, Confirm BF16 `n = 32`):** Canonical scoring awards fractional credit (`k / 4`) on 4-value `ruler_mv_niah` prompts. Replacing fractional credit with strict binary exact match (`1.0` only when all 4 values match) preserves falsification: `B1 - B4_H2O_MiKV = -0.0769 [-0.1875, +0.0108]`, and `B1 - B5_Random = +0.1131 [-0.0069, +0.2477]`.

### 3.3 Full Arm Comparison Across Held-Out Packs

| Arm | Prelim FP16 All-23 Score | Prelim FP16 Primary-20 Score | Prelim FP16 TF KL | Confirm BF16 All-36 Score | Confirm BF16 Primary-32 Score | Confirm BF16 TF KL | Mean b/e (Confirm) |
| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `C0_ActT` | `0.6474` | `0.5945` | `0.000000` | `0.5409` | `0.4836` | `0.000000` | `16.0000` |
| `C1_FP8` | `0.6471` | `0.5941` | `0.002292` | `0.5147` | `0.4540` | `0.003561` | `8.0000` |
| `A1_Q4` | `0.6164` | `0.5589` | `0.005911` | `0.4605` | `0.3931` | `0.005471` | `4.5000` |
| `A2_Q4_Hopper*` | `0.6904` | `0.6440` | `0.005148` | `0.5343` | `0.4761` | `0.004871` | `4.5010` |
| `A3_HGMP_Stab` | `0.7136` | `0.6707` | `0.018379` | `0.5347` | `0.4765` | `0.028598` | `4.5000` |
| `A6_H2O_MiKV` | `0.5837` | `0.5213` | `0.016763` | `0.5163` | `0.4559` | `0.025628` | `4.5000` |
| `B0_Q2_g64` | `0.5336` | `0.4636` | `0.048579` | `0.3765` | `0.2985` | `0.055155` | `2.5007` |
| `B1_HGMP_Stab` | `0.5401` | `0.4711` | `0.023155` | `0.4869` | `0.4228` | `0.027314` | `3.2500` |
| `B2_HGMP_Drift` | `0.4404` | `0.4565` | `0.034365` | `0.4335` | `0.3627` | `0.036421` | `3.2500` |
| `B3_HGMP_KDev` | `0.6415` | `0.5877` | `0.026579` | `0.4644` | `0.3975` | `0.014938` | `3.2500` |
| `B4_H2O_MiKV` | `0.6264` | `0.5704` | `0.013582` | `0.5344` | `0.4762` | `0.024785` | `3.2500` |
| `B5_Random` (3-seed) | `0.5993` | `0.5725` | `0.031081` | `0.3795` | `0.3019` | `0.038652` | `3.2500` |
| `B6_Oracle_MiKV` | `0.6387` | `0.5845` | `0.004764` | `0.5114` | `0.4504` | `0.007628` | `3.2500` |
| `B7_ColdSketch` | `0.0539` | `0.0620` | `0.760422` | `0.0349` | `0.0392` | `0.755419` | `2.6000` |
| `B7_Rehydrated_Top25pct` | `0.6108` | `0.5524` | `0.201950` | `0.5765` | `0.5235` | `0.196960` | `5.7240` |
| `C_Ledger_Pinned` (Ledger) | `1.0000` (`n=3`) | n/a | `0.022248` | `1.0000` (`n=4`) | n/a | `0.024589` | `3.2500` |

`*A2_Q4_Hopper` is labeled an over-budget comparator in Phase 0 because per-visit key means add `+118,848` to `+127,488 B` (Prelim) and `+118,784` to `+178,112 B` (Confirm) above `A1_Q4`.

### 3.4 Regime A (`A2_Q4_Hopper`) and Phase 1 Gate 1 Dev Calibration

In Phase 0, `A2_Q4_Hopper` improved over `A1_Q4` on both held-out packs (note that the 95% score CIs cross zero while teacher-forced KL drops consistently):

- **Prelim FP16:** all-23 paired `A2 - A1 = +0.0740 [-0.0013, +0.1922]`; primary-20 `+0.0851 [-0.0015, +0.2203]`; TF KL `0.005148` vs `0.005911` (`-12.9%`).
- **Confirm BF16:** all-36 paired `A2 - A1 = +0.0738 [-0.0088, +0.1760]`; primary-32 `+0.0830 [-0.0098, +0.1883]`; TF KL `0.004871` vs `0.005471` (`-11.0%`), with `A2` matching `C0_ActT` on `ruler_mk_niah` (`1.0000` vs `A1` `0.7500`) and `ruler_vt` (`0.2500` vs `A1` `0.0000`).

Phase 0 `A2_Q4_Hopper` computed the key mean across the entire prefix (non-causal). To select a strictly prefix-causal production policy without inspecting held-out data, we ran Phase 1 Gate 1 on `phase0a_dev_16k_32k` in BF16 on Devbox RTX PRO 6000 (`job-1126115112-8a86b1`, `att-c4003b4e`, `C0_ActT` self-KL `0.0` across all 576 positions):

| Gate 1 Candidate (Dev BF16, `n = 16` primary / `n = 18` all) | Primary TF KL | All-18 TF KL | Primary Top-1 (`512` pos) | Effective b/e | External Calibration |
| :--- | ---: | ---: | ---: | ---: | :--- |
| `A1_Q4` (stock Q4_0 baseline) | `0.003502` | `0.004107` | `497/512` (`97.07%`) | `4.5000` | None |
| **`M1_CausalW512_s4` (FROZEN WINNER)** | **`0.002851`** | **`0.003103`** | **`501/512` (`97.85%`)** | `4.5022` to `4.5045` | **None** |
| `M2_StaticDevCal_s0` | `0.003238` | `0.003909` | `495/512` (`96.68%`) | `4.5002` to `4.5005` | Dev calibration file |
| `M1_CausalW512_s0` | `0.003373` | `0.003783` | `497/512` (`97.07%`) | `4.5002` to `4.5005` | None |
| `M2_StaticDevCal_s4` | `0.003459` | `0.003647` | `495/512` (`96.68%`) | `4.5022` to `4.5045` | Dev calibration file |
| `M0_WhtOnly_s4` | `0.003576` | `0.003689` | `502/512` (`98.05%`) | `4.5020` to `4.5040` | None |
| `M0_WhtOnly_s0` | `0.003740` | `0.004378` | `498/512` (`97.27%`) | `4.5000` | None |
| `M3_PerPage4096_s4` | `0.003797` | `0.004142` | `502/512` (`98.05%`) | `4.5041` to `4.5064` | None |
| `M3_PerPage4096_s0` | `0.004200` | `0.004870` | `498/512` (`97.27%`) | `4.5020` to `4.5024` | None |

`M1_CausalW512_s4` (strictly causal `W = 512` key centering plus 4 full-precision `ActT` prefix sink tokens, zero external calibration) won outright, cutting primary TF KL by `18.6%` relative to `A1_Q4` and outperforming both `M0_WhtOnly_s4` (`+0.000725` higher KL) and `M1_CausalW512_s0` (`+0.000522` higher KL).

### 3.5 Regime C (Verified-Ledger Pinning)

- **Prelim FP16 (`n = 3`, 32K):** `C_Ledger_Pinned = 1.0000` (`3/3`), compared with `B5_Random` mean `0.7778` (`1.0, 0.3333, 1.0`) and `B0_Q2_g64 = 1.0000`.
- **Confirm BF16 (`n = 4`, 64K to 128K):** `C_Ledger_Pinned = 1.0000` (`4/4`), tying unpinned `B0_Q2_g64` (`4/4`) and all three `B5_Random` seeds (`4/4`).

The Phase 0 Verified-Ledger prompts were too easy to separate pinned from unpinned compression at 64K to 128K. For Phase 1 Gate 3, we built and froze an independent 24-prompt stress pack (`phase1_ledger_stress_64k_256k.json`, SHA-256 `0f55124ca93dc1b5e7a1b76eb7468be0e55ee4f5e7c37fa8c592d8d9508ea08a`, `att-e354e62c`) spanning 64K to 256K with equal-count `DisplacedPin` controls separated by at least 256 tokens.

### 3.6 H4 Deployed Cold-Sketch Rehydration (`B7`) and Separate 1M Pack

| Metric | Prelim FP16 (32K to 64K) | Confirm BF16 (32K to 256K) | Separate 1M Pack BF16 (`3/3` Complete) |
| :--- | :--- | :--- | :--- |
| Items observed (eligible / blind) | `23` (`20` eligible, `3` blind) | `36` (`36` eligible, `0` blind) | `3` (`3` eligible, `0` blind) |
| Cold pages / needle cold pages | `193` / `33` | `886` / `64` | `772` / `3` |
| Page AUROC (95% CI) | `0.8553 [0.7398, 0.9464]` | `0.7809 [0.6763, 0.8750]` | `0.9961 [0.9961, 0.9962]` |
| Recall@1 (95% CI) | `0.3500 [0.1625, 0.5500]` | `0.2458 [0.1264, 0.3778]` | `0.0000 [0.0000, 0.0000]` |
| Recall@4 (`>= 16` cold pages) | `n = 0` (none eligible) | `0.7250 [0.5000, 0.9125]` (`n = 16`) | `1.0000 [1.0000, 1.0000]` (`n = 3`) |
| Recall@10% (95% CI) | `0.5000 [0.3000, 0.7000]` | `0.5542 [0.4028, 0.7056]` | `1.0000 [1.0000, 1.0000]` |
| Recall@25% (95% CI) | `0.7250 [0.5500, 0.8750]` | `0.6556 [0.5028, 0.7972]` | `1.0000 [1.0000, 1.0000]` |
| Random expected Recall@25% | `0.2819` | `0.2702` | `0.2513` |
| Paired score delta (`B7_Reh - B7_Cold`) | `+0.5569 [+0.3633, +0.7494]` | `+0.5416 [+0.3829, +0.6986]` | `+0.6667 [0.0000, +1.0000]` |
| Paired TF KL delta (`B7_Reh - B7_Cold`) | `-0.5585 [-0.7525, -0.3706]` | `-0.5585 [-0.7371, -0.3847]` | `-0.5697 [-1.0181, -0.1230]` |
| Active working set (`b/e`) | `2.60 -> 5.68` (`+7,646` tok / `+238.9 MiB`) | `2.60 -> 5.72` (`+20,982` tok / `+655.7 MiB`) | `2.60 -> 5.77` (`+209,707` tok / `+6.40 GiB`) |

**Per-Item 1M Breakdown (`phase0b_needle_1m`, `att-ae276a49`):**

| Item | Input Tokens (Prefix + Q) | Host / Scratch | `C0_ActT` Output | `A1_Q4` / `A2_Hopper` Score (KL) | `B7_Cold` -> `B7_Reh` Score (KL) | Needle Page Rank / Eligible Cold Pages |
| :--- | ---: | :--- | :--- | :--- | :--- | :--- |
| `needle_1m_d15` | `1,049,034` (`1,048,998 + 36`) | Vast PRO 6000 Max-Q, NVMe | Unmatched preamble (`0.0`) | `0.0` (`0.0076`) / `1.0` (`0.0073`) | `0.0` (`0.1598`) -> `1.0` (`0.0369`) | Rank 2 of `255` |
| `needle_1m_d50` | `1,049,322` (`1,049,286 + 36`) | Vast PRO 6000 Max-Q, NVMe | Unmatched preamble (`0.0`) | `0.0` (`0.0199`) / `0.0` (`0.0123`) | `0.0` (`0.6307`) -> `1.0` (`0.0628`) | Rank 2 of `255` |
| `needle_1m_d85` | `1,079,121` (`1,079,085 + 36`) | Colab G4 PRO 6000 Server, `/dev/shm` | `DECOY-178186-OMEGA` (`0.0`) | `0.0` (`0.0026`) / `0.0` (`0.0034`) | `0.0` (`1.0494`) -> `0.0` (`0.0313`, matched `C0` decoy) | Rank 2 of `262` |

On `needle_1m_d85`, `B7_Rehydrated_Top25pct` retrieved the true needle page at rank 2 of 262 cold pages (`page_auroc = 0.9962`), reduced KL from `1.0494` to `0.0313`, and reproduced the exact decoy token emitted by full-precision `C0_ActT` (`DECOY-178186-OMEGA`). On `d15` and `d50`, `C0_ActT` generated a search preamble rather than the direct key string, whereas `B7_Rehydrated_Top25pct` emitted the exact gold keys (`KRYPTON-177800-OMEGA` and `KRYPTON-177993-OMEGA`).

### 3.7 Hardware-Labeled Non-Overlapping Timings

Because `arms_total_seconds` in raw logs includes `h4_deployed_b7_seconds`, every row below separates `arms_excl_b7` (`arms_total_seconds - h4_deployed_b7_seconds`) from `h4_deployed_b7` so stage times sum cleanly without overlap.

| Run | Hardware, CPU Threads, Scratch | Items / Tokens | Load (s) | Prefill (s) | Snap (s) | Q-Cap (s) | Saliency (s) | H4 Sketch (s) | Arms excl. B7 (s) | H4 B7 (s) | Sum Items (s) |
| :--- | :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| Dev FP16 | V100-PCIE-32GB, `16t`, NVMe | `18` / `385,016` | `21.16` | `575.56` | `15.10` | `8.66` | `160.06` | `93.68` | `4,302.21*` | incl.* | `5,155.75` |
| Prelim FP16 | V100-PCIE-32GB, `16t`, NVMe | `23` / `915,174` | `20.24` | `1,571.65` | `32.74` | `15.28` | `381.25` | `167.69` | `2,649.58` | `239.93` | `5,058.84` |
| Confirm BF16 | Vast RTX PRO 6000 Max-Q (`62696540`), `32t` on `20` eff vCPU, NVMe | `36` / `3,849,935` | `14.63` | `504.91` | `54.93` | `2.61` | `1,310.09` | `533.76` | `2,937.62` | `434.70` | `5,780.28` |
| 1M `d15` | Vast RTX PRO 6000 Max-Q (`62696540`), `32t` on `20` eff vCPU, NVMe | `1` / `1,049,034` | `14.75` | `427.39` | `11.20` | `0.44` | `435.73` | `185.22` | `549.68` | `122.85` | `1,733.02` |
| 1M `d50` | Vast RTX PRO 6000 Max-Q (`62696540`), `32t` on `20` eff vCPU, NVMe | `1` / `1,049,322` | (shared) | `426.79` | `10.79` | `0.43` | `435.51` | `183.21` | `555.42` | `123.30` | `1,735.96` |
| 1M `d85` | Colab G4 RTX PRO 6000 Server (`fleet-colab-g4`), `32t` on `48` AMD EPYC 9B45 vCPU, `87G /dev/shm` | `1` / `1,079,121` | `11.03` | `383.90` | `7.86` | `0.44` | `186.53` | `89.39` | `320.97` | `62.63` | `1,052.20` |
| Gate 1 Dev BF16 | Devbox RTX PRO 6000 Workstation (`sm_120a`), `8t`, `8.0G` peak RAM, `0B` swap | `18` / `384,399` | `132.41` | `cal: 38.72` | n/a | n/a | n/a | n/a | `eval: 545.03` | n/a | `716.16` |

`*` The Dev FP16 run predated `B7_Rehydrated_Top25pct` timing separation (`eval_seconds = 5,157.07s`). Vast `d15`/`d50` timings are kept strictly separate from Colab G4 `d85` because host CPU count (`20` vs `48` vCPU), storage (`NVMe` vs `/dev/shm`), and GPU TDP differ.

**Failed and Stopped Runs Recorded Separately:**
1. `job-1126091531-ffd686` (Devbox local 1M `d85` attempt, `4t`, `/mnt/storage` HDD): stopped early at `1h36m` after `/proc/io` confirmed `126.6 MB/s` sustained HDD read on `35.36 GiB` snapshots (`~91m` left vs `24m` wall cap). Scratch directory cleaned (`33 GiB` removed); replaced by `fleet-colab-g4` (`17.5m`).
2. `job-1126112841-9b54bf` (Devbox Gate 1 initial attempt, `16t`, `MemoryMax 12288 MiB`): killed by OOM (`exit 9`, `12.0G` peak RAM, `0B` swap, `0` output files written) while holding 18 snapshots (`12.32 GiB`) simultaneously. Resolved by 1-item streaming in commit `177294fa` (`job-1126115112-8a86b1`, `8.0G` peak, `exit 0`).

---

## 4. Distinctions, Limits, and Engineering Decision

1. **Loop-pass homology fails to beat attention history:** Although `B1_HGMP_Stab` beats random tiering at 32K to 256K (`+12.09 pts`), it trails `B4_H2O_MiKV` in both FP16 (`-9.92 pts`) and BF16 (`-5.34 pts`; `-7.69 pts` under strict all-4 MV scoring) and never reaches `0.75` needle AUROC (`0.6744`, `0.6810`). We drop H1/H2 homology tiering from production.
2. **H4 `B7` qualifiers must remain explicit:**
   - Question query vectors in Phase 0 were captured on the uncompressed `ActT` prefix (`reference-conditioned`), not on a compressed cold prefix.
   - `B7_Rehydrated_Top25pct` missed the `0.90` Recall@25% bar at 32K to 64K (`0.7250`) and 32K to 256K (`0.6556`), reaching `1.00` only on the 3-item 1M single-needle pack.
   - Rehydration expands active KV memory from `2.60 b/e` to `5.68` to `5.77 b/e` (`+238.9 MiB` at 32K to 64K, `+655.7 MiB` at 32K to 256K, `+6.40 GiB` at 1M), and Phase 0 did not include an equal-byte random-page-fetch control.
3. **Regime C ceiling effect:** `4/4` on Confirm BF16 tied unpinned `B0_Q2_g64` and `B5_Random`; real ledger-pinning validation requires the frozen 24-ID stress pack (`0f55124c`) with equal-count `DisplacedPin` controls.
4. **Two-phase engineering plan:**
   - **Phase 1 (active):** Ship portable CUDA `M1_CausalW512_s4` Hopper Q4 (`kv_format = 3`, strictly causal `W = 512` centering, 4 real `ActT` prefix sink tokens, zero external calibration) plus Verified-Ledger `ActT` pinning and lossless `HQ4S2` `INGCKPT1` checkpointing inside `atlas-ingine`.
   - **Phase 2 (queued behind Phase 1):** Implement true CUDA block-sparse Top-K KV decode on top of Phase 1 Hopper Q4 with strictly unconditioned live queries, an equal-byte random-page-fetch control, and position-encoding ablation at 64K to 1M.

---

## 5. Artifact Ledger, Compute Spend, and Attestation Receipts

### 5.1 Authoritative Artifacts

| Artifact | Path | SHA-256 |
| :--- | :--- | :--- |
| Phase 0 binary (`sm_120a`) | `/mnt/storage/hgmp_kv_880_bin/eval_hgmp_kv_phase0_sm120a_fast` | `4fc0ff56fa1b8ab6e8d848b3c1ab0b63ec921fe35c905fc4960d2062e4d9397b` |
| Gate 1 binary (`sm_120a`) | `/mnt/storage/hgmp_kv_880_bin/eval_hopper_q4_gate1_sm120a` | `37d1ed240f238aae973302bba058ca1990a5062846234916d53a34b4c5eb259b` |
| Dev FP16 raw (`18/18`) | `/mnt/storage/hgmp_kv_880_results/phase0a_dev_16k_32k_sm70.json` | `61c8c2cc76b101bea66a69f91d9d67fc1da6bd9842a26b2382f62c3d3be1ffd0` |
| Frozen direction JSON | `/mnt/storage/hgmp_kv_880_results/phase0_frozen_direction.json` | `a6f83b7b538169452a7bdb59b97dc1912e07623eaf4a6126154d9dec6f25daca` |
| Prelim FP16 raw (`23/23`) | `/mnt/storage/hgmp_kv_880_results/phase0a_prelim_32k_64k_sm70.json` | `50d72a740a5d54d43bd381c444c624268266678d051284c9902f45d30f043903` |
| Prelim FP16 scored | `/mnt/storage/hgmp_kv_880_results/phase0a_prelim_sm70_scored_stability.json` | `6b9f8271ca20647ef072a55508e53ddb08048486878a2594c5ca84548922bc7a` |
| Confirm BF16 raw (`36/36`) | `/mnt/storage/hgmp_kv_880_results/vast_shards/hgmp880-p0b-pro6000-20261126-r2/phase0b_confirm_32k_256k_bf16.json` | `0240e94ab2bc8a95ac30c6e8647e38cee4a514af206657bc990d729722a9b3a0` |
| Confirm BF16 scored | `/mnt/storage/hgmp_kv_880_results/phase0b_confirm_bf16_scored_stability.json` | `0d87ef3a9bea2bf69d1c94c9110f390a279c71c0aaeb5b089de4708f3d197f11` |
| 1M Vast immutable shard (`d15, d50`) | `/mnt/storage/hgmp_kv_880_results/vast_shards/hgmp880-p0b-pro6000-20261126-r2/phase0b_needle_1m_bf16.json` | `74406d627dbbb7467a5071dbaced4c93286bf7decb000f1b9d9f282d7e9b2a17` |
| 1M Colab G4 row (`d85`) | `/mnt/storage/hgmp_kv_880_results/phase0b_needle_1m_bf16_item2_3_colab_g4.json` | `32443f1ce936117abbeb536e9f133019bf9d16c6654de743676a7d172924ab16` |
| 1M merged raw (`3/3`) | `/mnt/storage/hgmp_kv_880_results/phase0b_needle_1m_bf16_complete_3items.json` | `89578b08b51d0f1e076a4e974678ea6ebdce7967b9c5cea7f2298d50dd2fcf41` |
| 1M scored (`3/3`) | `/mnt/storage/hgmp_kv_880_results/phase0b_needle_1m_bf16_scored_stability.json` | `7004b37d56c4077181546a32208835858368220ed7514a817b44ad2295c051b6` |
| Gate 1 Dev BF16 raw (`18/18`) | `/mnt/storage/hgmp_kv_880_results/phase1_gate1_dev_bf16.json` | `583bfc435bc2cdad951083c9884bec93ca961a164fcce14654b4c010016edf08` |
| Gate 1 Frozen Policy | `/mnt/storage/hgmp_kv_880_results/phase1_gate1_frozen_policy.json` | `fac3d4757431dd5b5789cbda612ce16d18c5b6390f477778bc8ad8d5da5ad15c` |
| Gate 3 Ledger Stress Pack (`24` IDs) | `/mnt/storage/hgmp_kv_880_packs/phase1_ledger_stress_64k_256k.json` | `0f55124ca93dc1b5e7a1b76eb7468be0e55ee4f5e7c37fa8c592d8d9508ea08a` |

### 5.2 External Compute Spend and Attestation Receipts

- **Vast.ai (`#885`):** `$4.541` (`contract 62696540`) + `$0.004` (`contract 62695086`) = **`$4.545` total** (`att-f4d0e87c`, confirmed `att-94a24102`, zero remaining instances).
- **Google Colab G4 (`fleet-colab-g4`):** Pre-paid CU balance logged at `2,470.36 CU` (11:27:41 PST) and `2,460.80 CU` after `d85` completion across two live G4 sessions (`~5.6 CU` attributable to `d85` 17.5m run plus staging; session warm-handed back to `agent-24`).
- **Attestation Receipts:** `att-048619a3` (dev direction freeze), `att-01d0b3c9` (prelim FP16), `att-21045c83` (independent prelim verification), `att-baf4e8ec` (confirm BF16), `att-82a19df1` (independent confirm verification + strict MV sensitivity), `att-f4d0e87c` (Vast 2-row 1M + billing), `att-94a24102` (independent Vast verification), `att-ae276a49` (3/3 1M merge + score), `att-1036e665` (Phase 1 Gate 0 CPU oracle), `att-e354e62c` (Gate 3 frozen 24-ID ledger stress pack), `att-c4003b4e` (Phase 1 Gate 1 dev calibration).

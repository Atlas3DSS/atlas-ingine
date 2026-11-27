# Physical GPU benchmarks

Dataset snapshot: 2026-11-27. These 25 measured physical GPU models span Pascal, Turing, Ampere, Ada, and Blackwell.
They include consumer, workstation, and cloud cards. Names and capacities are physical `nvidia-smi` readings.
For example, the L4 reports 23,034 MiB; provider-advertised capacity is not substituted.

## Method and source

Checkpoint: [Atlas3D/Inkling-2-Mini-atlas-exl3-9g-fp8-ingine](https://huggingface.co/Atlas3D/Inkling-2-Mini-atlas-exl3-9g-fp8-ingine/tree/6a118e83ac566f4b908f777e27a4bb9afc35a12e), revision `6a118e83ac566f4b908f777e27a4bb9afc35a12e`.

`ingine-bench` measures built-in hash-map chat and LRU-cache code prompts with `--effort medium --tokens 512 --runs 2`.
Decode uses a 16,384-token or 65,536-token KV allocation. A separate pass measures synthetic 16,000-token or 64,000-token prefill.
The decode numbers therefore do not measure generation after consuming those long synthetic prompts.
The two code revisions and the residency/offload configuration are recorded per GPU below.
The three 6 GB cards run plain decoding with `--atlas-6g`; drafting is unsupported on that tier.

The source is `consolidated_911_bundle.json`, SHA-256 `0297d1e5c9146b98e85aec5415f8ff0351b38451511155397a98d23ca27be87e`.
Its original per-run hardware and benchmark digests are retained with the dataset.
The [website data](https://monkeyalignment.com/data/inkling_engine.json), under `consumer_gpu_911`, provides the public projection of this same source.
The [engine page](https://monkeyalignment.com/inkling/engine/#consumer-gpu-sweep) presents the measurements with their limits.
Archived Volta, RTX PRO 6000, Loktar, NAND2RISCV, and persona examples on that page have separate hardware attribution.
They are not results from these Vast rentals.

## Hardware and configuration

| Physical GPU | VRAM (MiB) | Compute capability | Build architecture | Tier / offload mode | Code revision |
| --- | --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 5090 | 32,607 | 12.0 | `sm120a_bf16` | `resident_20g_plus` / `none` | `d1e221d6` |
| NVIDIA GeForce RTX 5080 | 16,303 | 12.0 | `sm120a_bf16` | `resident_16g` / `none` | `d1e221d6` |
| NVIDIA RTX PRO 4500 Blackwell | 32,623 | 12.0 | `sm120a_bf16` | `resident_20g_plus` / `none` | `d1e221d6` |
| NVIDIA GeForce RTX 5070 Ti | 16,303 | 12.0 | `sm120a_bf16` | `resident_16g` / `none` | `d1e221d6` |
| NVIDIA GeForce RTX 4080 | 16,376 | 8.9 | `sm89_bf16` | `resident_16g` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 4070 Ti SUPER | 16,376 | 8.9 | `sm89_bf16` | `resident_16g` / `none` | `9dc5d910` |
| NVIDIA RTX PRO 4000 Blackwell | 24,467 | 12.0 | `sm120a_bf16` | `resident_20g_plus` / `none` | `d1e221d6` |
| NVIDIA GeForce RTX 5070 | 12,227 | 12.0 | `sm120a_bf16` | `resident_11g_12g` / `none` | `d1e221d6` |
| NVIDIA GeForce RTX 4070 | 12,282 | 8.9 | `sm89_bf16` | `resident_11g_12g` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 5060 Ti | 16,311 | 12.0 | `sm120a_bf16` | `resident_16g` / `none` | `d1e221d6` |
| NVIDIA GeForce RTX 4060 Ti | 16,380 | 8.9 | `sm89_bf16` | `resident_16g` / `none` | `9dc5d910` |
| Quadro RTX 8000 | 49,152 | 7.5 | `sm75_fp16` | `resident_20g_plus` / `none` | `9dc5d910` |
| NVIDIA RTX A4000 | 16,376 | 8.6 | `sm86_bf16` | `resident_16g` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 3060 | 12,288 | 8.6 | `sm86_bf16` | `resident_10g_12g` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 2060 | 12,288 | 7.5 | `sm75_fp16` | `resident_10g_12g` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 3080 | 10,240 | 8.6 | `sm86_bf16` | `offload_10g` / `tiered_moe_vram8400` | `9dc5d910` |
| Tesla T4 | 15,360 | 7.5 | `sm75_fp16` | `resident_16g` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 3060 Ti | 8,192 | 8.6 | `sm86_bf16` | `offload_8g` / `tiered_moe_vram7200` | `9dc5d910` |
| NVIDIA GeForce RTX 3070 | 8,192 | 8.6 | `sm86_bf16` | `offload_8g` / `tiered_moe_vram7200` | `9dc5d910` |
| NVIDIA L4 | 23,034 | 8.9 | `sm89_bf16` | `resident_20g_plus` / `none` | `9dc5d910` |
| NVIDIA GeForce RTX 5060 | 8,151 | 12.0 | `sm120a_bf16` | `offload_8g` / `tiered_moe_vram7200` | `d1e221d6` |
| NVIDIA RTX A2000 | 6,138 | 8.6 | `sm86_bf16` | `offload_6g` / `atlas_6g_vram5888` | `9dc5d910` |
| NVIDIA GeForce GTX 1080 Ti | 11,264 | 6.1 | `sm61_fp16` | `resident_10g_12g` / `none` | `9dc5d910` |
| NVIDIA GeForce GTX 1660 | 6,144 | 7.5 | `sm75_fp16` | `offload_6g` / `atlas_6g_vram5888` | `9dc5d910` |
| NVIDIA GeForce RTX 3050 | 6,144 | 8.6 | `sm86_bf16` | `offload_6g` / `atlas_6g_vram5888` | `9dc5d910` |

Revisions: `9dc5d91088e72b076b5d331c08d236ac05bbecf9` and `d1e221d62b5889f1c826655bd6b8c6cfbdddd122`.
The tier names are recorded configurations, not a claim that all workloads fit inside physical VRAM.

## Short-prompt decode

All rates are tokens/s, chat / code. All 25 models have a 16K-allocation measurement; 23 have a 64K-allocation measurement.

| Physical GPU | nvidia-smi VRAM (MiB) | 16K allocation: chat / code | 64K allocation: chat / code, --spec | 64K allocation: chat / code, drafting off |
| --- | --- | --- | --- | --- |
| NVIDIA GeForce RTX 5090 | 32,607 | 484.48 / 617.78 (`--spec`) | 484.63 / 618.00 | 203.60 / 203.60 |
| NVIDIA GeForce RTX 5080 | 16,303 | 283.54 / 361.68 (`--spec`) | 283.12 / 360.83 | 158.33 / 158.34 |
| NVIDIA RTX PRO 4500 Blackwell | 32,623 | 259.44 / 327.70 (`--spec`) | 258.34 / 326.41 | 145.82 / 145.40 |
| NVIDIA GeForce RTX 5070 Ti | 16,303 | 226.02 / 331.63 (`--spec`) | 223.69 / 331.17 | 137.56 / 138.47 |
| NVIDIA GeForce RTX 4080 | 16,376 | 216.22 / 321.43 (`--spec`) | 216.16 / 321.23 | 132.86 / 132.97 |
| NVIDIA GeForce RTX 4070 Ti SUPER | 16,376 | 197.61 / 202.26 (`--spec`) | 197.51 / 202.16 | 107.16 / 107.19 |
| NVIDIA RTX PRO 4000 Blackwell | 24,467 | 173.02 / 246.27 (`--spec`) | 171.59 / 249.13 | 113.27 / 112.31 |
| NVIDIA GeForce RTX 5070 | 12,227 | 168.72 / 247.84 (`--spec`) | 168.54 / 247.34 | 124.95 / 124.95 |
| NVIDIA GeForce RTX 4070 | 12,282 | 150.51 / 220.24 (`--spec`) | 150.00 / 219.57 | 93.36 / 93.48 |
| NVIDIA GeForce RTX 5060 Ti | 16,311 | 142.42 / 192.83 (`--spec`) | 142.29 / 192.88 | 101.73 / 101.71 |
| NVIDIA GeForce RTX 4060 Ti | 16,380 | 108.00 / 115.77 (`--spec`) | 107.92 / 115.65 | 61.37 / 61.38 |
| Quadro RTX 8000 | 49,152 | 89.51 / 135.36 (`--spec`) | 89.63 / 135.45 | 79.89 / 79.88 |
| NVIDIA RTX A4000 | 16,376 | 84.01 / 109.51 (`--spec`) | 83.16 / 108.09 | 61.19 / 60.46 |
| NVIDIA GeForce RTX 3060 | 12,288 | 69.69 / 83.83 (`--spec`) | 69.62 / 83.83 | 48.43 / 48.23 |
| NVIDIA GeForce RTX 2060 | 12,288 | 59.23 / 71.34 (`--spec`) | 58.88 / 71.05 | 39.13 / 39.04 |
| NVIDIA GeForce RTX 3080 | 10,240 | 49.80 / 51.87 (`--spec`) | 56.11 / 66.06 | 58.08 / 57.93 |
| Tesla T4 | 15,360 | 47.38 / 62.22 (`--spec`) | 46.64 / 61.37 | 36.55 / 36.43 |
| NVIDIA GeForce RTX 3060 Ti | 8,192 | 36.75 / 41.84 (`--spec`) | 30.70 / 35.25 | 39.93 / 40.48 |
| NVIDIA GeForce RTX 3070 | 8,192 | 36.52 / 43.38 (`--spec`) | 30.13 / 36.31 | 42.03 / 42.43 |
| NVIDIA L4 | 23,034 | 35.29 / 43.60 (`--spec`) | 34.17 / 42.57 | 24.47 / 24.04 |
| NVIDIA GeForce RTX 5060 | 8,151 | 27.63 / 33.87 (`--spec`) | 21.87 / 27.20 | 34.39 / 34.84 |
| NVIDIA RTX A2000 | 6,138 | 28.40 / 28.14 (drafting off) | Drafting unsupported | 26.59 / 26.73 |
| NVIDIA GeForce GTX 1080 Ti | 11,264 | 18.84 / 21.77 (`--spec`) | Unavailable (command rejected) | Unavailable (command rejected) |
| NVIDIA GeForce GTX 1660 | 6,144 | 10.66 / 10.54 (drafting off) | Drafting unsupported | Timeout |
| NVIDIA GeForce RTX 3050 | 6,144 | 10.75 / 10.85 (drafting off) | Drafting unsupported | 9.38 / 9.99 |

GTX 1080 Ti's primary 64K commands were rejected (exit 2, unsupported `--no-vision` flag).
The preserved sidecar recovered the 16K result, but both 64K retries timed out; no 64K rate was accepted.
GTX 1660's plain 64K run timed out after 1,200.37 seconds; its speculative mode is unsupported.
Unavailable cells are not zero-throughput measurements.

## Separate synthetic prefill

All rates are input tokens/s. The first column uses 16,000 synthetic tokens; the two 64K columns use 64,000.
These rates must not be presented as decode throughput or as measurements from the 262K/1M needle probes.

| Physical GPU | 16K prefill | 64K prefill, --spec | 64K prefill, drafting off |
| --- | --- | --- | --- |
| NVIDIA GeForce RTX 5090 | 9757.5 (`--spec`) | 8430.8 | 8426.7 |
| NVIDIA GeForce RTX 5080 | 5439.2 (`--spec`) | 4618.8 | 4651.7 |
| NVIDIA RTX PRO 4500 Blackwell | 5096.7 (`--spec`) | 4600.5 | 4573.0 |
| NVIDIA GeForce RTX 5070 Ti | 4584.9 (`--spec`) | 3980.7 | 3986.3 |
| NVIDIA GeForce RTX 4080 | 4516.6 (`--spec`) | 3920.3 | 3925.1 |
| NVIDIA GeForce RTX 4070 Ti SUPER | 3847.2 (`--spec`) | 3413.4 | 3420.8 |
| NVIDIA RTX PRO 4000 Blackwell | 3564.0 (`--spec`) | 3204.7 | 3160.5 |
| NVIDIA GeForce RTX 5070 | 2913.5 (`--spec`) | 2558.2 | 2560.9 |
| NVIDIA GeForce RTX 4070 | 2517.3 (`--spec`) | 2133.7 | 2137.9 |
| NVIDIA GeForce RTX 5060 Ti | 2380.0 (`--spec`) | 2065.3 | 2066.3 |
| NVIDIA GeForce RTX 4060 Ti | 2020.0 (`--spec`) | 1774.2 | 1777.9 |
| Quadro RTX 8000 | 1479.7 (`--spec`) | 1341.2 | 1336.7 |
| NVIDIA RTX A4000 | 1826.3 (`--spec`) | 1584.6 | 1566.7 |
| NVIDIA GeForce RTX 3060 | 1124.2 (`--spec`) | 975.4 | 977.3 |
| NVIDIA GeForce RTX 2060 | 660.8 (`--spec`) | 598.0 | 594.4 |
| NVIDIA GeForce RTX 3080 | 987.8 (`--spec`) | 1136.0 | 1218.5 |
| Tesla T4 | 571.8 (`--spec`) | 512.1 | 507.1 |
| NVIDIA GeForce RTX 3060 Ti | 756.2 (`--spec`) | 639.1 | 788.8 |
| NVIDIA GeForce RTX 3070 | 783.0 (`--spec`) | 654.7 | 816.9 |
| NVIDIA L4 | 755.2 (`--spec`) | 687.4 | 639.9 |
| NVIDIA GeForce RTX 5060 | 571.2 (`--spec`) | 482.1 | 641.9 |
| NVIDIA RTX A2000 | 646.0 (drafting off) | Drafting unsupported | 580.6 |
| NVIDIA GeForce GTX 1080 Ti | 150.8 (`--spec`) | Unavailable (command rejected) | Unavailable (command rejected) |
| NVIDIA GeForce GTX 1660 | 62.6 (drafting off) | Drafting unsupported | Timeout |
| NVIDIA GeForce RTX 3050 | 203.3 (drafting off) | Drafting unsupported | 194.2 |

## Long-context probes

Successful 262K probes used a 262,144-token target in a 1M allocation, with 262,048 fresh document tokens.
Successful 1M runs used a 1,048,576-token target with 1,048,480 fresh document tokens and 3/3 needles.
For the latter, `resumed_tokens=0`, `decode_ctx=1,048,504`, and `decode_tokens=64`.
Only the RTX 4070 Ti SUPER and RTX 3060 completed that 1M target.

The numeric cells below are prefill tokens/s / long-context decode tokens/s, followed by needle recall.
Four cards completed the 262K probe. The RTX 2060 and RTX 3070 deferred the 1M run because it would exceed the two-hour cap.
Waves using `--skip-1m` record `one_m_skipped_by_policy`; this is neither a failure nor a successful 1M test.

| Physical GPU | 262K target in 1M allocation | 1M target |
| --- | --- | --- |
| NVIDIA GeForce RTX 5090 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 5080 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA RTX PRO 4500 Blackwell | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 5070 Ti | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 4080 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 4070 Ti SUPER | 1867.7 / 17.18; 3/3 needles | 753.2 / 1.65; 3/3 needles |
| NVIDIA RTX PRO 4000 Blackwell | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 5070 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 4070 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 5060 Ti | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 4060 Ti | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| Quadro RTX 8000 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA RTX A4000 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 3060 | 622.5 / 6.42; 3/3 needles | 261.1 / 1.43; 3/3 needles |
| NVIDIA GeForce RTX 2060 | 445.2 / 5.52; 3/3 needles | Deferred: exceeds 2-hour cap |
| NVIDIA GeForce RTX 3080 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| Tesla T4 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 3060 Ti | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 3070 | 540.4 / 4.60; 3/3 needles | Deferred: exceeds 2-hour cap |
| NVIDIA L4 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce RTX 5060 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA RTX A2000 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |
| NVIDIA GeForce GTX 1080 Ti | Timeout | Not run after unsuccessful 262K probe |
| NVIDIA GeForce GTX 1660 | Timeout | Not run after unsuccessful 262K probe |
| NVIDIA GeForce RTX 3050 | `one_m_skipped_by_policy` | `one_m_skipped_by_policy` |

GTX 1080 Ti's 262K probe timed out after 1,803.19 seconds; GTX 1660's timed out after 1,800.39 seconds.
Their 1M runs did not proceed after those unsuccessful probes.

## Drafting on 8 GB cards

At 64K KV allocation, the three measured 8 GB cards were faster without speculative drafting:

| Physical GPU | Plain chat tokens/s | Speculative chat tokens/s |
| --- | --- | --- |
| NVIDIA GeForce RTX 3070 | 42.03 | 30.13 |
| NVIDIA GeForce RTX 3060 Ti | 39.93 | 30.70 |
| NVIDIA GeForce RTX 5060 | 34.39 | 21.87 |

Additional expert transfers across the roughly 7.2 GB VRAM/PCIe boundary are an unmeasured hypothesis.
These runs did not profile that mechanism; the measured result is the throughput reversal.

## Failed or aborted bookings

Seven other bookings remain in the source separately from the 25 measured GPU rows:

- Three never reached an SSH-ready state: requested RTX 4060, RTX 4070 Ti SUPER, and the first RTX 5090 booking.
- Two stopped during upload/authentication: requested RTX 5060 Ti and a second RTX 5070 Ti booking.
- An earlier RTX 3080 run rejected benchmark arguments and failed its 262K probe with an allocation error.
- An earlier RTX 5070 Ti run rejected benchmark arguments and failed its 262K probe with an invalid CUDA cluster configuration.

Requested SKU names in this list are not physical hardware verification for bookings that never became accessible.
Later successful bookings are separate records. Short-prompt throughput alone does not establish output-quality parity across cards.

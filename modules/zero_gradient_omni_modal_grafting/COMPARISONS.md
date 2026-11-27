# Side-by-Side Visual Comparisons: Native Text Encoder vs. Zero-Gradient Plato's Key vs. B0 Shuffled Control

Every contact sheet below is a **byte-identical, SHA-256-verified** side-by-side comparison rendered on real GPUs with **zero upstream diffusion text encoder loaded** in the Plato's Key and matched deranged `B0` control arms (`COMPARISONS_MANIFEST.sha256.json`). Both winning transfers and explicit failure boundaries (`SDXL`, `turnaround_360_v4` strict gate `FAIL`) are published transparently.

- **Main Paper (`v0.9`):** [PAPER.md](PAPER.md)
- **Shot-by-Shot Walkthrough:** [TUTORIAL.md](TUTORIAL.md)
- **Host-Enforced Alignment & Live Pliny Proof:** [HOST_ENFORCED_ALIGNMENT.md](HOST_ENFORCED_ALIGNMENT.md)
- **Hugging Face Collection (`Omni by Atlas`):** https://huggingface.co/collections/Atlas3D/omni-by-atlas-6b09daccd0f65e23233163b3

---

## 1. Multi-Donor Packet A — MiniMax-H3 Sealed Confirm-160 (8 Disjoint Domains)

### Qwen3.8-27B → MiniMax-H3 (`concat4`, Sealed Confirm-160)
![Qwen3.8-27B to MiniMax-H3 Confirm-160 8-domain side-by-side](figures/comparisons/multidonor_qwen38_h3_confirm160_8domains.png)

### DeepSeek-V4.1-Flash → MiniMax-H3 (`concat4` `200 MiB`, `92.48%` Sealed Confirm-160, `16/16` `p = 3.05e-5`)
![DeepSeek-V4.1-Flash to MiniMax-H3 Confirm-160 8-domain side-by-side](figures/comparisons/multidonor_deepseek_v41_h3_confirm160_8domains.png)

---

## 2. Wave A — Paired Base & Turbo Image Models on Blackwell (`sm_120`, Zero Upstream Text Encoder)

### Z-Image Turbo (`98.88%` SigLIP2 Retention, `45.1 MiB` key, `-8.10 GB` VRAM)
![Z-Image Turbo Blackwell side-by-side](figures/comparisons/waveA_z_image_turbo_blackwell.png)

### Z-Image Base (`101.19%` SigLIP2 Retention, `28` steps CFG `4.0`)
![Z-Image Base Blackwell side-by-side](figures/comparisons/waveA_z_image_base_blackwell.png)

### FLUX.2-klein-4B Turbo (`97.73%` SigLIP2 Retention, `136 MiB` key)
![FLUX.2-klein-4B Turbo Blackwell side-by-side](figures/comparisons/waveA_flux2_klein_4b_turbo_blackwell.png)

### FLUX.2-klein-4B Base (`83.49%` SigLIP2 Retention, `28` steps CFG `4.0`)
![FLUX.2-klein-4B Base Blackwell side-by-side](figures/comparisons/waveA_flux2_klein_4b_base_blackwell.png)

### Qwen-Image-2.1 Turbo (`96.37%` SigLIP2 Retention, zero `Qwen3-VL-8B` loaded, `16.33 GB` peak VRAM)
![Qwen-Image-2.1 Turbo Blackwell side-by-side](figures/comparisons/waveA_qwen_image_21_turbo_blackwell.png)

### Qwen-Image-2.1 Base (`98.80%` SigLIP2 Retention, zero `Qwen3-VL-8B` loaded, `16.33 GB` peak VRAM)
![Qwen-Image-2.1 Base Blackwell side-by-side](figures/comparisons/waveA_qwen_image_21_base_blackwell.png)

---

## 3. Wave B — Universal Image Models on Blackwell (Dev-24 & Sealed Confirm-160 Word Bar)

### FLUX.2-dev (`99.65%` Native Retention — `271 MiB` key replacing `44.7 GiB` `Mistral-Small-3.1-24B`)
![FLUX.2-dev Blackwell side-by-side](figures/comparisons/waveB_flux2_dev_blackwell.png)

### p_image_oss (`99.09%` Dev-24; Sealed Confirm-160 `cos = 0.9746`, `R^2 = 0.8449` PASS)
![p_image_oss Blackwell side-by-side](figures/comparisons/waveB_p_image_oss_blackwell.png)

### moss_image Base (`98.25%` Dev-24; Sealed Confirm-160 `cos = 0.9191`, `R^2 = 0.6869` PASS)
![moss_image Base Blackwell side-by-side](figures/comparisons/waveB_moss_image_base_blackwell.png)

### moss_image Turbo (`95.38%` Dev-24)
![moss_image Turbo Blackwell side-by-side](figures/comparisons/waveB_moss_image_turbo_blackwell.png)

### FLUX.2-klein-9B Base (`97.17%` Dev-24; Sealed Confirm-160 `cos = 0.9426`, `R^2 = 0.7197` PASS)
![FLUX.2-klein-9B Base Blackwell side-by-side](figures/comparisons/waveB_flux2_klein_9b_base_blackwell.png)

### FLUX.1-dev (`93.07%` Dev-24 PASS; Confirm-160 word bar `R^2 = 0.5296`, below `0.55` threshold)
![FLUX.1-dev Blackwell side-by-side](figures/comparisons/waveB_flux1_dev_blackwell.png)

### FLUX.1-schnell (`91.44%` Dev-24 PASS)
![FLUX.1-schnell Blackwell side-by-side](figures/comparisons/waveB_flux1_schnell_blackwell.png)

---

## 4. Explicit Published Failure Boundaries & Work-in-Progress Runs

### SDXL Base & Turbo — Negative Control (`CLIP-L` Unmasked Padding Boundary: `57.69%` / `42.66%` FAIL)
![SDXL Base negative control](figures/comparisons/waveB_sdxl_base_negative_control.png)
![SDXL Turbo negative control](figures/comparisons/waveB_sdxl_turbo_negative_control.png)

### 360° World-Model Permanence (`turnaround_360_v4` — Visual Room Return vs. Strict Gate `FAIL`, `v5` Queued)
Top row (`LEDGER-OFF`) suffers catastrophic forgetting (`"LAWCHTE SKOROAFAIT GIEK"`, `0%` plaque); bottom row (`LEDGER-ON`) returns to the Monkey Founder portrait and readable 24-word plaque at `341°–360°`. Automated strict gate honestly reported `FAIL` on mid-segment keyframe jumps (`f383`/`f762`) and uncropped full-frame Tesseract OCR; `v5` (`smooth-trajectory-v5` + blinded plaque-ROI RapidOCR) is queued while `v1–v4` remain preserved.
![360 turnaround v4 side-by-side contact sheet](figures/comparisons/turnaround_360_v4_contact_sheet.png)

---

## 5. Live Pliny Adversarial Red-Team & Zero-Chrome Neuron Lens × Verified Ledger UX

`Inkling-2-Mini-atlas-12g` (`--persona-probe-mode`): **Arm A (Constitution Only)** breached `22 / 40` (`55.0%`), **Arm B (Host Verified Ledger)** committed `0 / 40` (`0.0%`) with `10 / 10` (`100.0%`) benign utility and `3 / 3` `/v1/probe` `d=2027` HTTP `400` anti-oracle rejections, with live `7`-visit MoE expert routing telemetry stored directly on the hash-chained SQLite ledger.

![Zero-Chrome Neuron Lens x Verified Ledger UX](adversarial_ledger_proof/live_results_v2/zero_chrome_neuron_lens_ux_preview.png)

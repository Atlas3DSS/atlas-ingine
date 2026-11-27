# Start Here: The 5-Minute Plain-English Guide to Omni by Atlas

New to this repo? You do **not** need a PhD, a 96 GB GPU, or a giant training cluster to understand or use this. Here is the whole idea in plain English, followed by copy-paste steps you can run right now.

---

## What is a "Plato's Key" (Explain Like I'm 5)?

Imagine your local LLM (`Inkling-2-Mini`, `Qwen`, `DeepSeek`, `GLM`, `Ling`) is a smart brain that already understands words, stories, and scenes.

Now imagine you want to generate a picture or a video+audio clip (like `MiniMax-H3`, `FLUX.2-dev`, or `Z-Image`). Normally, those media models force you to load a **second huge 27 GB to 44 GB text encoder** into your GPU just to read your prompt. Most home GPUs run out of VRAM right there.

**A Plato's Key is a tiny universal translator plug (22 MB to 271 MB).**
- You run your prompt through your local LLM **once**.
- You multiply your LLM's internal thoughts by **one flat matrix** (`R`).
- That flat matrix translates your LLM's thoughts directly into the exact language the picture, video, music, or voice model expects.
- **Zero gradient training. Zero edits to your LLM weights. The 27 GB / 44 GB text encoder is completely gone from VRAM.**

Because it is just one math equation (closed-form ridge regression), anyone can build a new key for a new model on a normal CPU in about 6 minutes using 640 prompts.

---

## Step 1 (30 Seconds, Any Laptop, No GPU Needed): Build Your First Toy Plato's Key

Clone the repo and run our 40-line starter script. It only needs `python3` and `numpy`:

```bash
git clone https://github.com/Atlas3DSS/atlas-ingine.git
cd atlas-ingine
python3 examples/make_my_first_key.py
```

You will see it:
1. Take thoughts from Model A and Model B on 640 prompts.
2. Solve **one** closed-form equation (`R = (X_c^T X_c + lam I)^-1 X_c^T Y_c`) with zero training loops.
3. Test on 160 brand-new unseen prompts (`cos ~ 0.99` vs a shuffled control `B0` that fails).

Open [examples/make_my_first_key.py](examples/make_my_first_key.py) and read it. That single 10-line function `fit_platos_key()` is literally the exact mathematical engine behind the whole paper!

---

## Step 2 (5 Minutes, Any NVIDIA GPU 6 GB to 96 GB): Run Local `atlas-ingine`

Want to run `Inkling-2-Mini` locally fast (works on 10 years of NVIDIA cards, from a 2016 Tesla P100 and 12 GB RTX 3060/5070 up to RTX 3090/4090/5090/PRO 6000)?

```bash
git clone https://github.com/Atlas3DSS/atlas-ingine.git
cd atlas-ingine
./atlas-setup
```

`./atlas-setup` inspects your GPU, picks the right model pack for your VRAM (`9 GB`, `10 GB`, `12 GB`, or `20 GB`), builds the native CUDA engine, and starts a local OpenAI-compatible server + chat UI.

---

## Step 3: Grab Ready-Made Plato's Keys From Hugging Face (Zero 27 GB Text Encoder)

All verified Apache-2.0 keys live in one collection:
- **Hugging Face Collection:** https://huggingface.co/collections/Atlas3D/omni-by-atlas

Pick the model you want:
- **Inkling-2-Mini keys** (MiniMax-H3 Image + Video/Audio `90 MiB`, Z-Image, FLUX, Qwen-Image): `Atlas3D/Inkling-2-Mini-Omni-Platos-Keys`
- **Qwen3.8-27B keys** (including half-depth `layer_31` `50 MiB` key): `Atlas3D/Qwen3.8-27B-Omni-Platos-Keys`
- **Ling-3.2-air keys** (`100 MiB` `97.90%` Sealed Confirm-160): `Atlas3D/Ling-3.2-air-Omni-Platos-Keys`

Download any key with the Hugging Face CLI:

```bash
pip install huggingface_hub safetensors numpy
hf download Atlas3D/Inkling-2-Mini-Omni-Platos-Keys --local-dir ./platos_keys
```

Drop-in Python loaders for **Diffusers** and **ComfyUI** are included right in the repo at:
- [modules/zero_gradient_omni_modal_grafting/loaders/diffusers_loader.py](modules/zero_gradient_omni_modal_grafting/loaders/diffusers_loader.py)
- [modules/zero_gradient_omni_modal_grafting/loaders/comfyui_loader.py](modules/zero_gradient_omni_modal_grafting/loaders/comfyui_loader.py)
- [modules/zero_gradient_omni_modal_grafting/loaders/platos_key_schema.py](modules/zero_gradient_omni_modal_grafting/loaders/platos_key_schema.py)

Run the loader self-test on CPU:

```bash
python3 -m unittest discover -s modules/zero_gradient_omni_modal_grafting/loaders -p 'test_*.py'
```

---

## Step 4: How to Fit Your Own Plato's Key for ANY New Model

Want to connect your favorite LLM (**Model A**) to your favorite image/video/audio generator (**Model B**)? Here is the entire recipe:

1. **Collect 640 diverse prompts** (keep 160 extra prompts completely sealed in a separate folder to test honestly at the end!).
2. **Extract aligned word vectors:**
   - Run the 640 prompts through Model A and grab 4 hidden layers concatenated together (`concat4`), mean-pooled per word.
   - Run the same 640 prompts through Model B's original text encoder and grab its conditioning embeddings, mean-pooled per word.
   - *(If using `Inkling-2-Mini`, zero-mask sink channel `d_2027 = 0`.)*
3. **Solve ONE FP64 ridge regression on CPU** (exact same math as `fit_platos_key()` in `examples/make_my_first_key.py`, picking `lambda` via inner cross-validation). Takes ~5–6 minutes on CPU and **0 GPU hours**.
4. **Save `R`, `mu_X`, and `mu_Y` to FP16 `.safetensors`**, unload Model B's giant text encoder forever, and feed `mu_Y + (x - mu_X) @ R` directly into Model B's cross-attention!

**Honest rule of thumb on what works vs what does not:**
- **Works great:** Image diffusion (`MiniMax-H3`, `FLUX.2-dev`, `Z-Image`, `Qwen-Image`, `FLUX.2/3-Klein`, `Krea2`), native Video+Audio (`MiniMax-H3 FL2VA/Ref2VA`, zero extra bytes!), music conditioning (`YuE2-3B`), and speech prosody/emotion (`Qwen3-TTS`).
- **Does NOT work linearly (we show the exact failures in the paper):** Reading tiny dense text/OCR (`Vision -> LLM`), raw continuous word spelling in TTS (supply discrete text tokens for exact words + Plato's Key for emotion/style), and unmasked 77-token `CLIP-L` (`SDXL`).

---

## Step 5: Cool Things to Explore Next

- **See the side-by-side pictures first (23 verified sheets):** [COMPARISONS.md](modules/zero_gradient_omni_modal_grafting/COMPARISONS.md)
- **Reproduce the 360° World-Model Turnaround ("This Is Not a Video"):** [WORLD_MODEL_TURNAROUND.md](modules/zero_gradient_omni_modal_grafting/world_model_turnaround/WORLD_MODEL_TURNAROUND.md)
- **Read the illustrated walkthrough:** [TUTORIAL.md](modules/zero_gradient_omni_modal_grafting/TUTORIAL.md)
- **Read the full paper (`v0.9.3`):** [PAPER.md](modules/zero_gradient_omni_modal_grafting/PAPER.md) or [PAPER.pdf](modules/zero_gradient_omni_modal_grafting/PAPER.pdf)
- **See Host-Enforced Verified Ledger Alignment (`0/40` breaches vs `22/40` prompt-only):** [HOST_ENFORCED_ALIGNMENT.md](modules/zero_gradient_omni_modal_grafting/HOST_ENFORCED_ALIGNMENT.md)
- **See Zero-Chrome Agentic UX (`Alpha` Suite — Blender, Godot, DragonFruit, Neuron Lens):** [ZERO_CHROME_AGENTIC_UX.md](docs/ZERO_CHROME_AGENTIC_UX.md)

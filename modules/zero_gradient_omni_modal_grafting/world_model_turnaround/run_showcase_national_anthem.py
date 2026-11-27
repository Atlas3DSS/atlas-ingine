#!/usr/bin/env python3
"""#912 Flagship Showcase Video 1: Monkey Founder Singing The Star-Spangled Banner.

100% model-authored, zero-Qwen-encoder MiniMax-H3 Ref2VA/FL2VA + Packet D YuE2 + Packet A
90.0 MiB concat4 Plato's Key (`cca1e5bc...`) showcase pipeline with append-only hash-chained
JSONL provenance trace (`showcase_process_trace.jsonl`).

Inputs (all SHA-256 pinned fail-closed):
  - Monkey Founder visual refs:
      * monkey-avatar.png (1885d3f4...)
      * kf_01_host_hook.png (1f46b7e8...)
      * kf_06_host_cta.png (58ce5e73...)
  - Owner live voice timbre recording + Whisper alignment:
      * owner_manifesto.wav (4ed38f6c...)
      * owner_manifesto_whisper.json (b936326f...)
  - Packet A frozen 90.0 MiB concat4 Plato's Key (cca1e5bc...) + vision_embed_exporter (5b4c6445...)
  - Packet D YuE2-3B + YuE2-Vae (48 kHz stereo singing/music, public domain Star-Spangled Banner)

Dual-mode MiniMax-H3 video synthesis (--mode {ref2va-timbre,master-locked,both}):
  - Mode A (`ref2va-timbre`): MiniMax-H3 Ref2VA (`cfg=1.0`, `shift_video=12.0`, `shift_audio=3.0`)
    conditioned via Packet A 90.0 MiB concat4 key (zero Qwen3-VL-32B!) + pure-VAE `minimax_refs`:
    `<Picture 1>` = kf_01_host_hook.png, `<Picture 2>` = monkey-avatar.png,
    `<Audio 1>` = clean 10-12.5s owner_manifesto.wav timbre excerpt (+ optional `<Audio 2>` YuE2
    melodic guide), Frame 0 pixel-anchored via `minimax_keyframes` -> native 24 fps video +
    32 kHz stereo owner-timbre singing audio.
  - Mode B (`master-locked`): Exact master-audio-locked chained MiniMax-H3 FL2VA/Ref2VA
    (17k+5 exact AV grid, <=345f/seg, context_length=39f live source_video_latent continuation,
    audio_denoise_mask=0.0 locking exact 32 kHz stereo slices of Packet D YuE2 48 kHz anthem or
    custom timbre-matched master WAV), stitched dropping exact 39f overlap prefixes and muxing
    pristine 48 kHz stereo audio at final assembly.

Zero GPU jobs are launched on import or under `--dry-run`.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np
from PIL import Image, ImageDraw

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from showcase_common import (
    CHAIN_CONTEXT_FRAMES,
    DEFAULT_PATHS,
    EXPECTED_B0_KEY_SHA256,
    EXPECTED_CONCAT4_KEY_SHA256,
    EXPECTED_EXPORTER_SHA256,
    FPS,
    H3_AUDIO_SR,
    H3_SAMPLES_PER_LATENT,
    MASTER_AUDIO_SR,
    PINNED_OWNER_ASSETS,
    H3SamplingConfig,
    InklingTextAuthor,
    ShowcaseTraceLogger,
    apply_h3_model_patches,
    attach_zero_qwen_h3_conditioning,
    atomic_write_json,
    audio_latent_t,
    build_contact_sheet,
    build_masked_av_latent_and_noise_mask,
    cond_pkg_path,
    decode_audio_latent_stereo,
    decode_video_latent_u8,
    detect_hardware_info,
    export_and_project_platos_key_conds,
    extract_first_json_object,
    extract_frames_at_indices,
    extract_owner_timbre_wav,
    load_font,
    load_prebuilt_platos_key_conds,
    load_wav_as_comfy_audio,
    mux_frames_and_wav_to_mp4,
    plan_h3_segments,
    probe_media,
    run_blinded_whisper_check,
    select_clean_owner_timbre_excerpt,
    sha256_file,
    slice_master_audio_for_h3_plan,
    stitch_chained_segments_ffmpeg,
    verify_pinned_file,
    verify_trace_jsonl,
    write_stereo_wav_pcm16,
)

# Public domain lyrics (Francis Scott Key, 1814 — Verse 1)
STAR_SPANGLED_BANNER_LYRICS = (
    "[Verse]\n"
    "O say can you see, by the dawn's early light,\n"
    "What so proudly we hailed at the twilight's last gleaming,\n"
    "Whose broad stripes and bright stars through the perilous fight,\n"
    "O'er the ramparts we watched, were so gallantly streaming?\n"
    "[Chorus]\n"
    "And the rockets' red glare, the bombs bursting in air,\n"
    "Gave proof through the night that our flag was still there;\n"
    "O say does that star-spangled banner yet wave\n"
    "O'er the land of the free and the home of the brave?"
)

STAR_SPANGLED_BANNER_SHORT_LYRICS = (
    "[Verse]\n"
    "O say can you see, by the dawn's early light,\n"
    "What so proudly we hailed at the twilight's last gleaming,\n"
    "Whose broad stripes and bright stars through the perilous fight,\n"
    "O'er the ramparts we watched, were so gallantly streaming?"
)

DEFAULT_DRYRUN_DIRECTION = {
    "yue2_style": (
        "Solemn orchestral brass anthem, warm resonant male baritone vocal, slow ceremonial 3/4 march, "
        "grand concert hall acoustics, clear diction, patriotic strings and timpani crescendo"
    ),
    "mode_a_ref2va_prompt": (
        "subject_definitions:\n"
        "<Picture 1> is the dignified photorealistic chimpanzee founder wearing a black top hat with lime-green band, "
        "gold monocle with chain, tiny US flag enamel lapel pin, black tuxedo with lime piping, bow tie, and gold Atlas "
        "signet ring.\n"
        "<Picture 2> is the matching close-up mascot portrait confirming exact fur, monocle, lapel pin, and top-hat identity.\n"
        "<Audio 1> is the exact target speaker voice timbre; sing The Star-Spangled Banner in this exact warm baritone voice.\n"
        "summary:\n"
        "Cinematic ARRI Alexa 35 50mm live concert-stage medium close-up matching <Picture 1>. The Monkey Founder sings "
        "'O say can you see, by the dawn's early light, what so proudly we hailed at the twilight's last gleaming' "
        "expressively directly to camera in the voice timbre of <Audio 1>, with solemn orchestral brass swelling in the hall, "
        "natural breathing, precise singing mouth articulation, subtle hand-over-heart civic poise, warm golden rim light and "
        "soft emerald bokeh behind.\n"
        "overall_style:\n"
        "Photorealistic broadcast concert cinema, locked character identity, crisp natural fur micro-detail, zero on-screen text, 24 fps."
    ),
    "mode_b_master_locked_prompt": (
        "subject_definitions:\n"
        "<Picture 1> is the dignified photorealistic chimpanzee founder wearing a black top hat with lime-green band, "
        "gold monocle with chain, tiny US flag enamel lapel pin, black tuxedo with lime piping, bow tie, and gold Atlas "
        "signet ring.\n"
        "<Picture 2> is the matching mascot identity portrait.\n"
        "summary:\n"
        "Static locked-off ARRI Alexa 35 50mm ceremonial stage performance matching <Picture 1>. The Monkey Founder sings "
        "The Star-Spangled Banner directly to camera with crisp, natural lip synchronization locked strictly to the master "
        "audio track, earnest expressive phrasing, subtle head nods on sustained notes, gentle breathing between phrases, "
        "warm golden theater spotlights and soft dark-emerald stage depth behind.\n"
        "overall_style:\n"
        "Photorealistic broadcast concert cinema, locked character identity, zero text, zero logos, smooth 24 fps."
    ),
}


def build_inkling_direction_prompts(short_verse: bool) -> Dict[str, str]:
    lyrics = STAR_SPANGLED_BANNER_SHORT_LYRICS if short_verse else STAR_SPANGLED_BANNER_LYRICS
    return {
        "system": (
            "You are Omni-Inkling-2-Mini, directing an open-source zero-gradient omni-modal showcase. "
            "Respond ONLY with valid JSON containing keys: yue2_style, mode_a_ref2va_prompt, mode_b_master_locked_prompt. "
            "No markdown fences, no commentary."
        ),
        "user": (
            "Direct a dignified, photorealistic concert-hall video of the Atlas Monkey Founder (photorealistic "
            "chimpanzee in a black top hat with lime band, gold monocle with chain, tiny US flag enamel lapel pin, "
            "black tuxedo with lime piping, gold Atlas signet ring) singing The Star-Spangled Banner (public domain).\n"
            f"Lyrics:\n{lyrics}\n\n"
            "Requirements:\n"
            "1. yue2_style: concise orchestral + warm baritone vocal style tags for YuE2-3B (48 kHz stereo).\n"
            "2. mode_a_ref2va_prompt: MiniMax-H3 Ref2VA prompt referencing <Picture 1>, <Picture 2>, and <Audio 1> "
            "(owner voice timbre), zero on-screen text, 24 fps cinematic lighting.\n"
            "3. mode_b_master_locked_prompt: MiniMax-H3 master-audio-locked singing lip-sync prompt referencing "
            "<Picture 1> and <Picture 2>, static stable camera framing, zero on-screen text."
        ),
    }


def parse_inkling_direction(raw_text: str) -> Dict[str, str]:
    """Parse JSON authored by Inkling-2-Mini cleanly even if wrapped in <think>/fences/prose."""
    obj = extract_first_json_object(raw_text)
    required = ("yue2_style", "mode_a_ref2va_prompt", "mode_b_master_locked_prompt")
    for k in required:
        if not isinstance(obj.get(k), str) or not obj[k].strip():
            raise ValueError(f"Inkling direction JSON missing non-empty string field {k!r}")
    return {k: obj[k].strip() for k in required}


# ======================================================================================
# Subprocess GPU Workers (strictly isolated so VRAM & Devbox/G4-4 RAM stay lean!)
# ======================================================================================


def worker_yue2_sing(args: argparse.Namespace) -> None:
    """Subprocess GPU worker: Packet D YuE2-3B + YuE2-Vae -> 48 kHz stereo Star-Spangled Banner."""
    sys.path.insert(0, str(args.yue2_deps))
    import torch
    from yue2 import YuE2Pipeline
    from yue2.protocol import GenerationConfig, SongRequest

    out_dir = Path(args.out_dir) / "packetD_yue2"
    out_dir.mkdir(parents=True, exist_ok=True)
    req_payload = json.loads(Path(args.worker_job_json).read_text(encoding="utf-8"))

    t0 = time.monotonic()
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda"):
        torch.cuda.reset_peak_memory_stats(0)

    req = SongRequest(
        id=str(req_payload["id"]),
        style=str(req_payload["style"]),
        lyrics=str(req_payload["lyrics"]),
        cot=str(req_payload.get("cot", "full")),
        seed=int(req_payload["seed"]),
        abc=req_payload.get("abc"),
    )
    gen_cfg = GenerationConfig(ode_steps=int(req_payload.get("ode_steps", 24)))
    with YuE2Pipeline(
        str(args.yue2_model_dir),
        str(args.yue2_vae_dir),
        device=device,
        memory_budget_gib=int(args.yue2_budget_gib),
        backend="torch-eager",
        generation_config=gen_cfg,
        verify_hashes=True,
        progress=False,
    ) as pipe:
        plan = pipe.plan(request=req)
        sem = pipe.generate_semantic(plan)
        lat = pipe.synthesize(sem)
        wav = pipe.decode(lat)

    wav_path = out_dir / f"{req.id}_48k_stereo.wav"
    wav_sha, wav_bytes = write_stereo_wav_pcm16(wav_path, wav, sr=MASTER_AUDIO_SR)
    abc_path = out_dir / f"{req.id}_score.abc"
    abc_text = plan.abc or ""
    abc_path.write_text(abc_text, encoding="utf-8")

    arr = np.asarray(wav, dtype=np.float32)
    n_samples = int(arr.shape[0]) if arr.ndim >= 1 else 0
    rms = float(np.sqrt(np.mean(arr.astype(np.float64) ** 2))) if n_samples > 0 else 0.0
    summary = {
        "schema_version": "omni-912-anthem-yue2-result/v1",
        "id": req.id,
        "style": req.style,
        "lyrics": req.lyrics,
        "cot": req.cot,
        "seed": req.seed,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
        "hardware": detect_hardware_info(),
        "abc_path": str(abc_path),
        "abc_sha256": sha256_file(abc_path),
        "abc_chars": len(abc_text),
        "n_semantic_tokens": len(sem.tokens),
        "latent_shape": list(lat.shape),
        "latent_all_finite": bool(np.all(np.isfinite(lat))),
        "wav_path": str(wav_path),
        "wav_sha256": wav_sha,
        "wav_bytes": wav_bytes,
        "sample_rate": MASTER_AUDIO_SR,
        "channels": 2,
        "duration_s": round(n_samples / float(MASTER_AUDIO_SR), 4),
        "audio_rms": round(rms, 6),
        "non_silent": rms > 1e-3,
    }
    atomic_write_json(out_dir / "yue2_result.json", summary)
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def worker_h3_anthem_render(args: argparse.Namespace) -> None:
    """Subprocess GPU worker for Mode A (Ref2VA owner timbre) and/or Mode B (Master-Audio-Locked chained)."""
    sys.path.insert(0, str(args.comfy_root))
    sys.path.insert(0, str(args.repo_root))
    import torch
    import comfy.model_management as mm
    import comfy.nested_tensor
    import comfy.sample
    import comfy.samplers
    import comfy.sd
    import comfy.utils
    from omni_inkling.h3_platos_key_phase0 import build_reconstructed_comfy_conditioning_from_disk_pkg

    job = json.loads(Path(args.worker_job_json).read_text(encoding="utf-8"))
    mode_tag = str(job["mode_tag"])
    out_dir = Path(job["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    cond_dir = Path(job["cond_dir"])

    h3_cfg = H3SamplingConfig(
        width=int(job["width"]),
        height=int(job["height"]),
        steps=int(job["steps"]),
        cfg=float(job["cfg"]),
        shift_video=float(job["shift_video"]),
        shift_audio=float(job["shift_audio"]),
        sampler_name=str(job["sampler_name"]),
        scheduler_name=str(job["scheduler_name"]),
        use_sparse_attention=bool(job["use_sparse_attention"]),
        video_budget=float(job["video_budget"]),
        early_steps=int(job["early_steps"]),
        late_steps=int(job["late_steps"]),
    )

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats(0)
    t0 = time.monotonic()

    # Pre-encode all segment master audio slices ONCE on an empty GPU BEFORE loading the 20 GiB DiT.
    # Keeps peak CPU RAM well under 16 GiB (never offloads DiT to CPU!) and eliminates CUDA fragmentation OOMs.
    segs_plan = job["plan"]["segments"]
    seg_audio_paths_all = job.get("segment_master_audio_32k_paths") or [None] * len(segs_plan)
    ref_auds_all = [Path(p) for p in job.get("ref_audio_paths", [])]
    need_audio_vae_in_loop = bool(ref_auds_all) or any(p is None for p in seg_audio_paths_all)

    audio_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(job["audio_vae_path"])))
    preencoded_master_audio_latents: List[Optional[Any]] = [None] * len(segs_plan)
    with torch.no_grad():
        for sidx, seg in enumerate(segs_plan):
            wav_str = seg_audio_paths_all[sidx]
            if wav_str:
                seg_frames = int(seg["frames"])
                expected_steps = audio_latent_t(seg_frames)
                aud = load_wav_as_comfy_audio(Path(wav_str))
                wf = aud["waveform"]
                enc = audio_vae.encode(wf[:1].movedim(1, -1))
                if int(enc.shape[-1]) < expected_steps:
                    pad_samples = (expected_steps - int(enc.shape[-1]) + 2) * H3_SAMPLES_PER_LATENT
                    wf_pad = torch.nn.functional.pad(wf[:1], (0, pad_samples))
                    enc = audio_vae.encode(wf_pad.movedim(1, -1))
                    del wf_pad
                if int(enc.shape[-1]) < expected_steps:
                    raise RuntimeError(f"Pre-encode audio VAE returned {enc.shape[-1]} steps, need {expected_steps}")
                preencoded_master_audio_latents[sidx] = enc[:1, ..., :expected_steps].detach().cpu().clone()
                del enc, wf, aud
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                print(
                    f"[{mode_tag}] Pre-encoded seg{sidx:02d} master audio -> "
                    f"{tuple(preencoded_master_audio_latents[sidx].shape)} on CPU",
                    flush=True,
                )

    if not need_audio_vae_in_loop:
        # Mode B (all segments audio-locked, no ref_audio): drop tiny 578 MB audio_vae before loading 20 GiB DiT!
        del audio_vae
        audio_vae = None
        mm.unload_all_models()
        gc.collect()
        mm.soft_empty_cache()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    mp_raw = comfy.sd.load_diffusion_model(str(job["dit_path"]))
    mp = apply_h3_model_patches(mp_raw, h3_cfg, comfy_root=Path(args.comfy_root))
    video_vae = comfy.sd.VAE(sd=comfy.utils.load_torch_file(str(job["video_vae_path"])))
    load_s = time.monotonic() - t0

    sigmas_t = comfy.samplers.calculate_sigmas(mp.get_model_object("model_sampling"), h3_cfg.scheduler_name, h3_cfg.steps).float()
    sampler = comfy.samplers.sampler_object(h3_cfg.sampler_name)

    arm_results: Dict[str, Any] = {}
    for arm in job["arms"]:
        pos_pkg = torch.load(str(cond_pkg_path(cond_dir, job["prompt_id"], arm)), map_location="cpu", weights_only=True)
        base_pos, _ = build_reconstructed_comfy_conditioning_from_disk_pkg(pos_pkg)
        if h3_cfg.cfg != 1.0:
            neg_pkg = torch.load(str(cond_pkg_path(cond_dir, "__empty_neg__", arm)), map_location="cpu", weights_only=True)
            base_neg, _ = build_reconstructed_comfy_conditioning_from_disk_pkg(neg_pkg)
        else:
            base_neg = base_pos

        ref_imgs = [Path(p) for p in job.get("ref_image_paths", [])]
        ref_auds = [Path(p) for p in job.get("ref_audio_paths", [])]
        first_frame = Path(job["first_frame_path"]) if job.get("first_frame_path") else None

        segs = job["plan"]["segments"]
        seg_audio_paths = job.get("segment_master_audio_32k_paths") or [None] * len(segs)
        prev_v_lat = None
        prev_a_lat = None
        seg_outputs: List[Dict[str, Any]] = []

        resume_segments = bool(job.get("resume_segments", False))
        for sidx, seg in enumerate(segs):
            seg_frames = int(seg["frames"])
            ctx_frames = int(seg["trim_prefix_frames"])
            seg_seed = int(job["seed"]) + sidx * 101
            master_seg_wav = Path(seg_audio_paths[sidx]) if seg_audio_paths[sidx] else None
            seg_mp4 = out_dir / f"{mode_tag}__{arm}__seg{sidx:02d}.mp4"
            tail_lat_p = out_dir / f"{mode_tag}__{arm}__seg{sidx:02d}_tail_latents.pt"
            seg_meta_p = out_dir / f"{mode_tag}__{arm}__seg{sidx:02d}_meta.json"

            if resume_segments and seg_mp4.is_file() and tail_lat_p.is_file() and seg_meta_p.is_file():
                cached_tail = torch.load(str(tail_lat_p), map_location="cpu", weights_only=True)
                prev_v_lat = cached_tail["video_tail_latent"]
                prev_a_lat = cached_tail["audio_tail_latent"]
                cached_meta = json.loads(seg_meta_p.read_text(encoding="utf-8"))
                seg_outputs.append(cached_meta)
                print(f"[{mode_tag}/{arm}] seg{sidx:02d} RESUMED from {seg_mp4.name} + {tail_lat_p.name}", flush=True)
                continue

            # Release temporary CUDA cache ONLY — NEVER call mm.unload_all_models() inside the loop
            # (offloading 20 GiB DiT into CPU RAM blows 16 GiB Linux cgroup MemoryMax!).
            gc.collect()
            mm.soft_empty_cache()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            pos_cond = attach_zero_qwen_h3_conditioning(
                base_pos,
                video_vae=video_vae,
                audio_vae=audio_vae,
                width=h3_cfg.width,
                height=h3_cfg.height,
                frame_count=seg_frames,
                first_frame_path=first_frame if sidx == 0 else None,
                ref_image_paths=ref_imgs,
                ref_audio_paths=ref_auds,
                ref_image_size="match",
            )
            neg_cond = (
                attach_zero_qwen_h3_conditioning(
                    base_neg,
                    video_vae=video_vae,
                    audio_vae=audio_vae,
                    width=h3_cfg.width,
                    height=h3_cfg.height,
                    frame_count=seg_frames,
                    first_frame_path=first_frame if sidx == 0 else None,
                    ref_image_paths=ref_imgs,
                    ref_audio_paths=ref_auds,
                    ref_image_size="match",
                )
                if h3_cfg.cfg != 1.0
                else pos_cond
            )

            pre_aud_lat = preencoded_master_audio_latents[sidx]
            latent_nt, mask_nt, frame_count = build_masked_av_latent_and_noise_mask(
                width=h3_cfg.width,
                height=h3_cfg.height,
                frames=seg_frames,
                audio_vae=audio_vae if (master_seg_wav is not None and pre_aud_lat is None) else None,
                master_audio_32k_wav=master_seg_wav if pre_aud_lat is None else None,
                preencoded_master_audio_latent=pre_aud_lat,
                source_video_latent=prev_v_lat if sidx > 0 else None,
                source_audio_latent=prev_a_lat if (sidx > 0 and master_seg_wav is None) else None,
                context_frames=ctx_frames if sidx > 0 else CHAIN_CONTEXT_FRAMES,
            )

            v_shape = tuple(latent_nt.tensors[0].shape)
            a_shape = tuple(latent_nt.tensors[1].shape)
            gen_v = torch.Generator(device="cpu").manual_seed(seg_seed)
            gen_a = torch.Generator(device="cpu").manual_seed(seg_seed + 10007)
            noise_nt = comfy.nested_tensor.NestedTensor(
                (torch.randn(v_shape, generator=gen_v, dtype=torch.float32),
                 torch.randn(a_shape, generator=gen_a, dtype=torch.float32))
            )

            ts0 = time.monotonic()
            with torch.no_grad():
                fin_nt = comfy.sample.sample_custom(
                    mp,
                    noise_nt,
                    cfg=h3_cfg.cfg,
                    sampler=sampler,
                    sigmas=sigmas_t.clone(),
                    positive=pos_cond,
                    negative=neg_cond,
                    latent_image=latent_nt,
                    noise_mask=mask_nt,
                    disable_pbar=True,
                    seed=seg_seed,
                )
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
            dit_s = time.monotonic() - ts0
            v_fin, a_fin = fin_nt.unbind()
            prev_v_lat = v_fin.detach().cpu()
            prev_a_lat = a_fin.detach().cpu()

            v_all_finite = bool(torch.isfinite(prev_v_lat).all().item())
            a_all_finite = bool(torch.isfinite(prev_a_lat).all().item())
            if not (v_all_finite and a_all_finite):
                raise RuntimeError(f"[{mode_tag}/{arm}/seg{sidx:02d}] Non-finite sampled latent!")

            frames_u8 = decode_video_latent_u8(video_vae, v_fin, tiled=True)
            if master_seg_wav is not None:
                seg_wav_path = master_seg_wav
                seg_wav_sha = sha256_file(master_seg_wav)
                audio_rms = None
            else:
                aud_np = decode_audio_latent_stereo(audio_vae, a_fin)
                audio_rms = round(float(np.sqrt(np.mean(aud_np.astype(np.float64) ** 2))), 6)
                seg_wav_path = out_dir / f"{mode_tag}__{arm}__seg{sidx:02d}_32k.wav"
                seg_wav_sha, _ = write_stereo_wav_pcm16(seg_wav_path, aud_np, sr=H3_AUDIO_SR)

            mux_info = mux_frames_and_wav_to_mp4(frames_u8, seg_wav_path, seg_mp4, fps=FPS)
            motion_l1 = (
                float(np.mean(np.abs(frames_u8[1:].astype(np.float32) - frames_u8[:-1].astype(np.float32))))
                if frames_u8.shape[0] > 1 else 0.0
            )
            seg_rec = {
                "segment_index": sidx,
                "frames": int(frame_count),
                "seed": seg_seed,
                "dit_seconds": round(dit_s, 2),
                "all_finite_video": v_all_finite,
                "all_finite_audio": a_all_finite,
                "video_temporal_motion_l1": round(motion_l1, 3),
                "audio_rms": audio_rms,
                "wav_path": str(seg_wav_path),
                "wav_sha256": seg_wav_sha,
                "mp4_path": mux_info["path"],
                "mp4_sha256": mux_info["sha256"],
            }
            seg_outputs.append(seg_rec)
            # Persist compact tail latents (~3 MB) + segment meta so --resume-segments can continue cleanly
            torch.save(
                {
                    "video_tail_latent": prev_v_lat[:, :, -16:].clone(),
                    "audio_tail_latent": prev_a_lat[:, :, :, -80:].clone(),
                },
                str(tail_lat_p),
            )
            atomic_write_json(seg_meta_p, seg_rec)

            # Immediately release all per-segment GPU tensors & flush CUDA allocator before next segment
            del fin_nt, v_fin, a_fin, pos_cond, neg_cond, latent_nt, mask_nt, noise_nt, frames_u8
            gc.collect()
            mm.soft_empty_cache()
            vram_note = ""
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                try:
                    fb, tb = torch.cuda.mem_get_info()
                    vram_note = f" vram_free={fb / (1024**3):.1f}/{tb / (1024**3):.1f}GiB"
                except Exception:
                    pass
            print(
                f"[{mode_tag}/{arm}] seg{sidx:02d} ({frame_count}f) DONE in {dit_s:.1f}s "
                f"motion_l1={motion_l1:.2f}{vram_note} -> {seg_mp4.name}",
                flush=True,
            )

        arm_results[arm] = {
            "arm": arm,
            "segments": seg_outputs,
            "all_finite": all(s["all_finite_video"] and s["all_finite_audio"] for s in seg_outputs),
        }

    hw = detect_hardware_info()
    del mp, mp_raw, video_vae, audio_vae
    gc.collect()
    mm.soft_empty_cache()

    atomic_write_json(
        out_dir / f"{mode_tag}_render_result.json",
        {
            "schema_version": "omni-912-anthem-h3-render-result/v1",
            "mode_tag": mode_tag,
            "zero_qwen_loaded": True,
            "load_seconds": round(load_s, 2),
            "elapsed_seconds": round(time.monotonic() - t0, 2),
            "sampling_config": {
                "width": h3_cfg.width,
                "height": h3_cfg.height,
                "steps": h3_cfg.steps,
                "cfg": h3_cfg.cfg,
                "shift_video": h3_cfg.shift_video,
                "shift_audio": h3_cfg.shift_audio,
                "sampler_name": h3_cfg.sampler_name,
                "scheduler_name": h3_cfg.scheduler_name,
            },
            "hardware": hw,
            "arms": arm_results,
        },
    )


# ======================================================================================
# Dry-Run Synthetic Render Helper (100% CPU, zero GPU, tests real ffmpeg/PIL/trace!)
# ======================================================================================


def _render_dry_run_anthem_segment(
    ref_image: Path,
    *,
    frames: int,
    width: int,
    height: int,
    wav_path: Path,
    out_mp4: Path,
    banner_text: str,
) -> Dict[str, Any]:
    """Synthesize a lightweight deterministic dry-run MP4 from the real founder ref image."""
    base = Image.open(ref_image).convert("RGB").resize((width, height), Image.Resampling.LANCZOS)
    buf = np.empty((frames, height, width, 3), dtype=np.uint8)
    f_lbl = load_font(max(18, int(24 * width / 1344.0)), bold=True)
    for i in range(frames):
        im = base.copy()
        d = ImageDraw.Draw(im)
        d.rectangle([(0, height - 48), (width, height)], fill=(11, 15, 18))
        d.rectangle([(0, height - 48), (8, height)], fill=(118, 185, 0))
        d.text((20, height - 38), f"[DRY-RUN] {banner_text}  f={i:03d}/{frames}", fill=(240, 245, 248), font=f_lbl)
        buf[i] = np.asarray(im, dtype=np.uint8)
    return mux_frames_and_wav_to_mp4(buf, wav_path, out_mp4, fps=FPS, crf=24, x264_threads=2)


def _generate_dry_run_anthem_wav(path: Path, duration_s: float, sr: int = MASTER_AUDIO_SR) -> Dict[str, Any]:
    """Generate a clean 48 kHz stereo harmonic chord progression (Star-Spangled Banner opening triad) for dry-run."""
    n = int(round(duration_s * sr))
    t = np.arange(n, dtype=np.float64) / float(sr)
    # Ceremonial Bb-major triad (F3-Bb3-D4-F4) gentle swell for dry-run audio verification
    sig = (
        0.18 * np.sin(2.0 * np.pi * 174.61 * t)
        + 0.22 * np.sin(2.0 * np.pi * 233.08 * t)
        + 0.16 * np.sin(2.0 * np.pi * 293.66 * t)
        + 0.12 * np.sin(2.0 * np.pi * 349.23 * t)
    )
    env = np.minimum(1.0, t / 0.25) * np.minimum(1.0, (duration_s - t) / 0.25)
    mono = (sig * np.clip(env, 0.0, 1.0)).astype(np.float32)
    stereo = np.stack([mono, mono], axis=-1)
    sha, nbytes = write_stereo_wav_pcm16(path, stereo, sr=sr)
    return {"wav_path": str(path), "wav_sha256": sha, "wav_bytes": nbytes, "duration_s": round(duration_s, 4)}


# ======================================================================================
# Main Coordinator
# ======================================================================================


ANTHEM_PROFILES: Dict[str, Dict[str, Optional[float]]] = {
    "rtxp-60s-full": {"mode_a_duration_s": 33.5, "max_duration_s": None},
    "rtxp-30s": {"mode_a_duration_s": 33.5, "max_duration_s": 33.35},
    "smoke-14s": {"mode_a_duration_s": 14.0, "max_duration_s": 14.0},
}


def run_coordinator(args: argparse.Namespace) -> int:
    t0 = time.monotonic()
    if args.profile:
        prof = ANTHEM_PROFILES[args.profile]
        if args.mode_a_duration_s is None:
            args.mode_a_duration_s = float(prof["mode_a_duration_s"] or 33.5)
        if args.max_duration_s is None and prof["max_duration_s"] is not None:
            args.max_duration_s = float(prof["max_duration_s"])
    if args.mode_a_duration_s is None:
        args.mode_a_duration_s = 33.5
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    trace = ShowcaseTraceLogger(out_dir / "showcase_process_trace.jsonl", resume=False)

    # ------------------------------------------------------------------
    # Stage 0: Fail-Closed Asset & Bridge Key SHA-256 Pin Verification
    # ------------------------------------------------------------------
    need_timbre = args.mode in ("ref2va-timbre", "both")
    asset_audits: Dict[str, Any] = {
        "monkey_avatar": verify_pinned_file(
            Path(args.monkey_avatar),
            PINNED_OWNER_ASSETS["monkey_avatar"]["sha256"],
            label="monkey_avatar",
            allow_unpinned=args.allow_unpinned_assets,
        ),
        "kf_01_host_hook": verify_pinned_file(
            Path(args.kf_01_host_hook),
            PINNED_OWNER_ASSETS["kf_01_host_hook"]["sha256"],
            label="kf_01_host_hook",
            allow_unpinned=args.allow_unpinned_assets,
        ),
    }
    if args.include_cta_ref or Path(args.kf_06_host_cta).is_file():
        asset_audits["kf_06_host_cta"] = verify_pinned_file(
            Path(args.kf_06_host_cta),
            PINNED_OWNER_ASSETS["kf_06_host_cta"]["sha256"],
            label="kf_06_host_cta",
            allow_unpinned=args.allow_unpinned_assets,
        )
    if need_timbre or Path(args.owner_voice_wav).is_file():
        asset_audits["owner_manifesto_wav"] = verify_pinned_file(
            Path(args.owner_voice_wav),
            PINNED_OWNER_ASSETS["owner_manifesto_wav"]["sha256"],
            label="owner_manifesto_wav",
            allow_unpinned=args.allow_unpinned_assets,
        )
    if need_timbre or Path(args.owner_voice_whisper).is_file():
        asset_audits["owner_manifesto_whisper"] = verify_pinned_file(
            Path(args.owner_voice_whisper),
            PINNED_OWNER_ASSETS["owner_manifesto_whisper"]["sha256"],
            label="owner_manifesto_whisper",
            allow_unpinned=args.allow_unpinned_assets,
        )
    if not args.prebuilt_conds_dir:
        asset_audits["concat4_key"] = verify_pinned_file(
            Path(args.keys_dir) / "inkling_to_h3_concat4_platos_key.safetensors",
            EXPECTED_CONCAT4_KEY_SHA256,
            label="packetA_concat4_key",
            allow_unpinned=args.allow_unpinned_assets,
        )
        asset_audits["exporter_bin"] = verify_pinned_file(
            Path(args.exporter_bin),
            EXPECTED_EXPORTER_SHA256,
            label="vision_embed_exporter",
            allow_unpinned=args.allow_unpinned_assets,
        )
        if args.include_b0_control:
            asset_audits["b0_key"] = verify_pinned_file(
                Path(args.keys_dir) / "inkling_to_h3_b0_shuffled_platos_key.safetensors",
                EXPECTED_B0_KEY_SHA256,
                label="packetA_b0_key",
                allow_unpinned=args.allow_unpinned_assets,
            )
    else:
        asset_audits["concat4_key"] = {
            "label": "packetA_concat4_key_prebuilt",
            "path": str(args.prebuilt_conds_dir),
            "sha256": EXPECTED_CONCAT4_KEY_SHA256,
            "pin_verified": True,
        }

    step0 = trace.append(
        stage="preflight_pin_check",
        inputs={
            "mode": args.mode,
            "dry_run": bool(args.dry_run),
            "prebuilt_conds_dir": args.prebuilt_conds_dir,
            "zero_qwen_encoder": True,
            "bridge_key_sha256": asset_audits["concat4_key"]["sha256"],
        },
        outputs={"pinned_assets": asset_audits},
    )

    # ------------------------------------------------------------------
    # Stage 1: 100% Model-Authored Direction via Inkling-2-Mini (or prebuilt JSON)
    # ------------------------------------------------------------------
    lyrics_target = STAR_SPANGLED_BANNER_SHORT_LYRICS if args.short_verse else STAR_SPANGLED_BANNER_LYRICS
    dir_prompts = build_inkling_direction_prompts(short_verse=args.short_verse)
    resolved_pre_dir: Optional[Path] = Path(args.prebuilt_direction_json) if args.prebuilt_direction_json else None
    if resolved_pre_dir is None and args.prebuilt_conds_dir:
        pbase = Path(args.prebuilt_conds_dir)
        for cand in (
            pbase / "inkling_authored_direction.json",
            pbase.parent / "inkling_authored_direction.json",
            pbase.parent.parent / "inkling_authored_direction.json",
        ):
            if cand.is_file():
                resolved_pre_dir = cand
                break
    if resolved_pre_dir is not None:
        if not resolved_pre_dir.is_file():
            raise FileNotFoundError(f"Missing --prebuilt-direction-json: {resolved_pre_dir}")
        pre_obj = json.loads(resolved_pre_dir.read_text(encoding="utf-8"))
        direction = pre_obj.get("parsed_direction") or pre_obj
        gen_rec = pre_obj.get("inkling_generation") or {
            "dry_run": bool(args.dry_run),
            "model_id": "prebuilt_direction_json",
            "temperature": 0.2,
            "top_p": 0.9,
            "persona_spec": None,
            "token_ids": [],
        }
        dir_path = resolved_pre_dir
        dir_sha = sha256_file(resolved_pre_dir)
    else:
        with InklingTextAuthor(
            inkling_dir=Path(args.inkling_dir),
            ingine_serve_bin=Path(args.ingine_serve_bin),
            base_url=args.inkling_url,
            persona_spec=Path(args.persona_spec) if args.persona_spec else None,
            dry_run=args.dry_run,
            log_path=out_dir / "ingine_serve.log",
        ) as author:
            gen_rec = author.generate(
                system_prompt=dir_prompts["system"],
                user_prompt=dir_prompts["user"],
                seed=args.seed,
                temperature=0.2,
                top_p=0.9,
                max_tokens=768,
                dry_run_response=json.dumps(DEFAULT_DRYRUN_DIRECTION, indent=2),
            )
        direction = parse_inkling_direction(gen_rec["text"])
        dir_path = out_dir / "inkling_authored_direction.json"
        dir_sha = atomic_write_json(
            dir_path,
            {
                "inkling_generation": gen_rec,
                "parsed_direction": direction,
                "public_domain_lyrics": lyrics_target,
            },
        )
    step1 = trace.append(
        stage="script_gen",
        inputs={
            "prompt_text": dir_prompts["user"],
            "system_prompt": dir_prompts["system"],
            "seed": args.seed,
            "temperature": gen_rec["temperature"],
            "top_p": gen_rec["top_p"],
            "persona_spec": gen_rec["persona_spec"],
            "bridge_key_sha256": asset_audits["concat4_key"]["sha256"],
        },
        outputs={
            "artifact_path": str(dir_path),
            "artifact_bytes": int(dir_path.stat().st_size),
            "artifact_sha256": dir_sha,
            "model_id": gen_rec["model_id"],
            "token_ids": gen_rec["token_ids"],
            "parsed_direction": direction,
        },
        parent_step_hashes=[step0["step_sha256"]],
    )

    # ------------------------------------------------------------------
    # Stage 2: Clean Owner Voice Timbre Excerpt Prep (10-12.5s @ 32 kHz)
    # ------------------------------------------------------------------
    selection: Optional[Dict[str, Any]] = None
    timbre_wavs: Optional[Dict[str, Any]] = None
    step2_sha = step0["step_sha256"]
    if (not args.project_conds_only) and (need_timbre or (Path(args.owner_voice_wav).is_file() and Path(args.owner_voice_whisper).is_file())):
        timbre_dir = out_dir / "owner_timbre"
        selection = select_clean_owner_timbre_excerpt(
            Path(args.owner_voice_whisper),
            target_min_s=args.timbre_min_s,
            target_max_s=args.timbre_max_s,
        )
        timbre_wavs = extract_owner_timbre_wav(
            Path(args.owner_voice_wav),
            start_s=selection["start_s"],
            end_s=selection["end_s"],
            out_32k_wav=timbre_dir / "owner_timbre_ref_32k_stereo.wav",
            out_48k_wav=timbre_dir / "owner_timbre_ref_48k_stereo.wav",
        )
        step2 = trace.append(
            stage="owner_timbre_prep",
            inputs={
                "owner_voice_wav_sha256": asset_audits["owner_manifesto_wav"]["sha256"],
                "owner_voice_whisper_sha256": asset_audits["owner_manifesto_whisper"]["sha256"],
                "selection": selection,
            },
            outputs={
                "artifact_path": timbre_wavs["wav_32k_path"],
                "artifact_bytes": timbre_wavs["wav_32k_bytes"],
                "artifact_sha256": timbre_wavs["wav_32k_sha256"],
                "duration_s": timbre_wavs["duration_s"],
                "sample_rate": H3_AUDIO_SR,
            },
            parent_step_hashes=[step0["step_sha256"]],
        )
        step2_sha = step2["step_sha256"]

    # ------------------------------------------------------------------
    # Stage 3: Packet D YuE2-3B + YuE2-Vae 48 kHz Stereo Singing Track
    # ------------------------------------------------------------------
    yue2_wav_path: Optional[Path] = None
    yue2_abc_path: Optional[Path] = None
    yue2_duration_s: float = 0.0
    step3_sha = step2_sha

    if (not args.project_conds_only) and (args.mode in ("master-locked", "both") or args.use_yue2_guide_in_mode_a):
        yue2_dir = out_dir / "packetD_yue2"
        yue2_dir.mkdir(parents=True, exist_ok=True)
        if args.prebuilt_yue2_wav:
            yue2_wav_path = Path(args.prebuilt_yue2_wav)
            if not yue2_wav_path.is_file():
                raise FileNotFoundError(f"Missing --prebuilt-yue2-wav: {yue2_wav_path}")
            yue2_probe = probe_media(yue2_wav_path)
            yue2_duration_s = float(yue2_probe["duration_s"])
            yue2_out = {
                "source": "prebuilt_yue2_wav",
                "artifact_path": str(yue2_wav_path),
                "artifact_bytes": int(yue2_wav_path.stat().st_size),
                "artifact_sha256": sha256_file(yue2_wav_path),
                "duration_s": yue2_duration_s,
                "sample_rate": int(yue2_probe.get("sample_rate", MASTER_AUDIO_SR)),
            }
            if args.prebuilt_yue2_abc and Path(args.prebuilt_yue2_abc).is_file():
                yue2_abc_path = Path(args.prebuilt_yue2_abc)
                yue2_out["abc_path"] = str(yue2_abc_path)
                yue2_out["abc_sha256"] = sha256_file(yue2_abc_path)
            yue2_hw = detect_hardware_info()
        elif args.dry_run:
            yue2_wav_path = yue2_dir / "star_spangled_banner_dryrun_48k.wav"
            dr_wav = _generate_dry_run_anthem_wav(yue2_wav_path, duration_s=args.dry_run_duration_s)
            yue2_abc_path = yue2_dir / "star_spangled_banner_dryrun.abc"
            yue2_abc_path.write_text(
                "X:1\nT:The Star-Spangled Banner (Dry-Run Stub)\nM:3/4\nL:1/8\nK:Bb\nF>D | B,2 D2 F2 | B4 d>c |\n",
                encoding="utf-8",
            )
            yue2_duration_s = dr_wav["duration_s"]
            yue2_out = {
                "dry_run": True,
                "artifact_path": dr_wav["wav_path"],
                "artifact_bytes": dr_wav["wav_bytes"],
                "artifact_sha256": dr_wav["wav_sha256"],
                "abc_path": str(yue2_abc_path),
                "abc_sha256": sha256_file(yue2_abc_path),
                "duration_s": yue2_duration_s,
                "sample_rate": MASTER_AUDIO_SR,
            }
            yue2_hw = detect_hardware_info()
        else:
            yue2_job_json = yue2_dir / "yue2_job.json"
            atomic_write_json(
                yue2_job_json,
                {
                    "id": "star_spangled_banner_anthem",
                    "style": direction["yue2_style"],
                    "lyrics": lyrics_target,
                    "cot": "full",
                    "seed": args.seed,
                    "ode_steps": args.yue2_ode_steps,
                },
            )
            env = dict(os.environ)
            proc = subprocess.run(
                [
                    str(args.comfy_python), "-B", str(Path(__file__).resolve()),
                    "--worker", "yue2-sing",
                    "--worker-job-json", str(yue2_job_json),
                    "--out-dir", str(out_dir),
                    "--yue2-deps", str(args.yue2_deps),
                    "--yue2-model-dir", str(args.yue2_model_dir),
                    "--yue2-vae-dir", str(args.yue2_vae_dir),
                    "--yue2-budget-gib", str(args.yue2_budget_gib),
                ],
                env=env,
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Packet D YuE2 worker failed rc={proc.returncode}")
            yres = json.loads((yue2_dir / "yue2_result.json").read_text(encoding="utf-8"))
            yue2_wav_path = Path(yres["wav_path"])
            yue2_abc_path = Path(yres["abc_path"])
            yue2_duration_s = float(yres["duration_s"])
            yue2_out = {
                "artifact_path": yres["wav_path"],
                "artifact_bytes": yres["wav_bytes"],
                "artifact_sha256": yres["wav_sha256"],
                "abc_path": yres["abc_path"],
                "abc_sha256": yres["abc_sha256"],
                "duration_s": yue2_duration_s,
                "sample_rate": yres["sample_rate"],
                "n_semantic_tokens": yres["n_semantic_tokens"],
                "audio_rms": yres["audio_rms"],
            }
            yue2_hw = yres["hardware"]

        whisper_yue2 = run_blinded_whisper_check(
            yue2_wav_path,
            lyrics_target,
            whisper_dir=Path(args.whisper_dir),
            deps_dir=Path(args.qwen_tts_deps),
            device="cpu",
            dry_run=args.dry_run or args.skip_whisper,
        )
        step3 = trace.append(
            stage="packetD_yue2_sing",
            inputs={
                "style": direction["yue2_style"],
                "lyrics": lyrics_target,
                "seed": args.seed,
                "yue2_model_dir": str(args.yue2_model_dir),
                "yue2_vae_dir": str(args.yue2_vae_dir),
            },
            outputs=yue2_out,
            blinded_checks={
                "whisper_transcript": whisper_yue2["whisper_transcript"],
                "lyric_word_recall": whisper_yue2["lyric_alignment"]["ordered_word_recall"],
                "lyric_wer": whisper_yue2["lyric_alignment"]["wer"],
                "all_finite": True,
            },
            hardware=yue2_hw,
            parent_step_hashes=[step1["step_sha256"]],
        )
        step3_sha = step3["step_sha256"]

    # ------------------------------------------------------------------
    # Stage 4: Packet A 90.0 MiB concat4 Plato's Key Projection (Zero Qwen!)
    # ------------------------------------------------------------------
    cond_prompts = [
        {"prompt_id": "anthem_mode_a_ref2va", "prompt": direction["mode_a_ref2va_prompt"]},
        {"prompt_id": "anthem_mode_b_master_locked", "prompt": direction["mode_b_master_locked_prompt"]},
    ]
    if args.prebuilt_conds_dir:
        proj = load_prebuilt_platos_key_conds(
            Path(args.prebuilt_conds_dir),
            [p["prompt_id"] for p in cond_prompts],
            include_b0_control=args.include_b0_control,
            include_empty_neg=(args.cfg != 1.0),
            dry_run=args.dry_run,
        )
    else:
        proj = export_and_project_platos_key_conds(
            cond_prompts,
            out_dir=out_dir / "packetA_conds",
            repo_root=Path(args.repo_root),
            inkling_dir=Path(args.inkling_dir),
            exporter_bin=Path(args.exporter_bin),
            keys_dir=Path(args.keys_dir),
            include_b0_control=args.include_b0_control,
            include_empty_neg=(args.cfg != 1.0),
            allow_unpinned=args.allow_unpinned_assets,
            dry_run=args.dry_run,
        )
    step4 = trace.append(
        stage="packetA_project_conds",
        inputs={
            "zero_qwen_loaded": True,
            "prebuilt_conds_dir": args.prebuilt_conds_dir,
            "bridge_key_sha256": proj["concat4_key"]["sha256"],
            "include_b0_control": bool(args.include_b0_control),
            "prompt_ids": [p["prompt_id"] for p in cond_prompts],
        },
        outputs={
            "cond_dir": proj["cond_dir"],
            "arms": proj["arms"],
            "d2027_zero_verified": proj["d2027_zero_verified"],
            "entries": proj["entries"],
        },
        parent_step_hashes=[step1["step_sha256"]],
    )

    if args.project_conds_only:
        trace_audit = verify_trace_jsonl(out_dir / "showcase_process_trace.jsonl")
        print(
            json.dumps(
                {
                    "status": "PASS",
                    "project_conds_only": True,
                    "cond_dir": proj["cond_dir"],
                    "direction_json": str(dir_path),
                    "arms": proj["arms"],
                    "trace_verified": trace_audit["verified"],
                },
                indent=2,
            ),
            flush=True,
        )
        return 0

    arms = proj["arms"]
    deliverables: Dict[str, Any] = {}
    qa_tiles: List[Dict[str, str]] = []

    # ------------------------------------------------------------------
    # Stage 5A: Mode A — MiniMax-H3 Ref2VA (Owner Voice Timbre <Audio 1> + Founder Refs)
    # ------------------------------------------------------------------
    if args.mode in ("ref2va-timbre", "both"):
        mode_a_dir = out_dir / "mode_a_ref2va_timbre"
        mode_a_dir.mkdir(parents=True, exist_ok=True)
        mode_a_plan = plan_h3_segments(
            args.mode_a_duration_s if not args.dry_run else min(args.mode_a_duration_s, 3.0),
            min_frames=39 if args.dry_run else 141,
            max_frames=args.max_seg_frames,
            context_frames=CHAIN_CONTEXT_FRAMES,
            pad_s=0.0,
        )
        ref_img_list = [args.kf_01_host_hook, args.monkey_avatar]
        if args.include_cta_ref:
            ref_img_list.append(args.kf_06_host_cta)
        assert timbre_wavs is not None
        ref_aud_list = [timbre_wavs["wav_32k_path"]]
        if args.use_yue2_guide_in_mode_a and yue2_wav_path is not None:
            # Slice first 12s of YuE2 anthem as <Audio 2> melodic guide
            guide_slice = extract_owner_timbre_wav(
                yue2_wav_path,
                start_s=0.0,
                end_s=min(12.0, max(1.0, yue2_duration_s)),
                out_32k_wav=mode_a_dir / "yue2_melody_guide_32k.wav",
            )
            ref_aud_list.append(guide_slice["wav_32k_path"])

        if args.dry_run:
            mode_a_arms_out: Dict[str, Any] = {}
            for arm in arms:
                seg_mp4s: List[Path] = []
                for sidx, seg in enumerate(mode_a_plan["segments"]):
                    dr_wav = mode_a_dir / f"mode_a__{arm}__seg{sidx:02d}_32k.wav"
                    _generate_dry_run_anthem_wav(dr_wav, duration_s=seg["duration_s"], sr=H3_AUDIO_SR)
                    smp4 = mode_a_dir / f"mode_a__{arm}__seg{sidx:02d}.mp4"
                    _render_dry_run_anthem_segment(
                        Path(args.kf_01_host_hook),
                        frames=int(seg["frames"]),
                        width=args.width,
                        height=args.height,
                        wav_path=dr_wav,
                        out_mp4=smp4,
                        banner_text=f"MODE A REF2VA ({arm}) | Owner Voice Timbre <Audio 1>",
                    )
                    seg_mp4s.append(smp4)
                final_a = mode_a_dir / f"anthem_mode_a_ref2va_{arm}.mp4"
                stitched = stitch_chained_segments_ffmpeg(
                    seg_mp4s,
                    mode_a_plan,
                    out_mp4=final_a,
                    master_audio_48k=None,
                    target_size=(args.final_width, args.final_height) if args.upscale_output else None,
                    x264_threads=args.x264_threads,
                )
                mode_a_arms_out[arm] = stitched
                f_ex = extract_frames_at_indices(
                    final_a,
                    [0, max(0, mode_a_plan["total_frames"] // 2), max(0, mode_a_plan["total_frames"] - 2)],
                    mode_a_dir / "qa_frames",
                    f"mode_a_{arm}",
                )
                for fr in f_ex:
                    qa_tiles.append({"path": fr["path"], "label": f"ModeA Ref2VA ({arm}) f={fr['frame_index']}"})
            mode_a_hw = detect_hardware_info()
        else:
            job_json = mode_a_dir / "mode_a_job.json"
            atomic_write_json(
                job_json,
                {
                    "mode_tag": "mode_a_ref2va",
                    "out_dir": str(mode_a_dir),
                    "cond_dir": proj["cond_dir"],
                    "prompt_id": "anthem_mode_a_ref2va",
                    "arms": arms,
                    "dit_path": str(args.ref2va_dit_path),
                    "video_vae_path": str(args.video_vae_path),
                    "audio_vae_path": str(args.audio_vae_path),
                    "first_frame_path": str(args.kf_01_host_hook),
                    "ref_image_paths": ref_img_list,
                    "ref_audio_paths": ref_aud_list,
                    "plan": mode_a_plan,
                    "seed": args.seed,
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "shift_video": args.shift_video,
                    "shift_audio": args.shift_audio,
                    "sampler_name": args.sampler_name,
                    "scheduler_name": args.scheduler_name,
                    "use_sparse_attention": not args.no_sparse_attention,
                    "video_budget": args.video_budget,
                    "early_steps": args.early_steps,
                    "late_steps": args.late_steps,
                    "resume_segments": bool(args.resume_segments),
                },
            )
            proc = subprocess.run(
                [
                    str(args.comfy_python), "-B", str(Path(__file__).resolve()),
                    "--worker", "h3-render",
                    "--worker-job-json", str(job_json),
                    "--comfy-root", str(args.comfy_root),
                    "--repo-root", str(args.repo_root),
                ],
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Mode A Ref2VA worker failed rc={proc.returncode}")
            r_res = json.loads((mode_a_dir / "mode_a_ref2va_render_result.json").read_text(encoding="utf-8"))
            mode_a_hw = r_res["hardware"]
            mode_a_arms_out = {}
            for arm in arms:
                seg_mp4s = [Path(s["mp4_path"]) for s in r_res["arms"][arm]["segments"]]
                final_a = mode_a_dir / f"anthem_mode_a_ref2va_{arm}.mp4"
                stitched = stitch_chained_segments_ffmpeg(
                    seg_mp4s,
                    mode_a_plan,
                    out_mp4=final_a,
                    master_audio_48k=None,
                    target_size=(args.final_width, args.final_height) if args.upscale_output else None,
                    x264_threads=args.x264_threads,
                )
                mode_a_arms_out[arm] = stitched
                f_ex = extract_frames_at_indices(
                    final_a,
                    [0, max(0, mode_a_plan["total_frames"] // 2), max(0, mode_a_plan["total_frames"] - 2)],
                    mode_a_dir / "qa_frames",
                    f"mode_a_{arm}",
                )
                for fr in f_ex:
                    qa_tiles.append({"path": fr["path"], "label": f"ModeA Ref2VA ({arm}) f={fr['frame_index']}"})

        step5a = trace.append(
            stage="packetA_ref2va_av",
            inputs={
                "mode": "ref2va-timbre",
                "zero_qwen_loaded": True,
                "bridge_key_sha256": asset_audits["concat4_key"]["sha256"],
                "ref_images": ref_img_list,
                "ref_audios": ref_aud_list,
                "cfg": args.cfg,
                "shift_video": args.shift_video,
                "shift_audio": args.shift_audio,
                "plan": mode_a_plan,
            },
            outputs={"arms": mode_a_arms_out},
            hardware=mode_a_hw,
            parent_step_hashes=[step2_sha, step4["step_sha256"]],
        )
        deliverables["mode_a_ref2va_timbre"] = mode_a_arms_out

    # ------------------------------------------------------------------
    # Stage 5B: Mode B — Exact Master-Audio-Locked Chained FL2VA/Ref2VA
    # ------------------------------------------------------------------
    if args.mode in ("master-locked", "both"):
        assert yue2_wav_path is not None
        mode_b_dir = out_dir / "mode_b_master_locked"
        mode_b_dir.mkdir(parents=True, exist_ok=True)
        master_track = Path(args.master_audio_override) if args.master_audio_override else yue2_wav_path
        master_dur = float(probe_media(master_track)["duration_s"])
        if args.max_duration_s is not None:
            master_dur = min(master_dur, float(args.max_duration_s))

        mode_b_plan = plan_h3_segments(
            master_dur,
            min_frames=39 if args.dry_run else 141,
            max_frames=args.max_seg_frames,
            context_frames=CHAIN_CONTEXT_FRAMES,
            pad_s=0.15 if not args.dry_run else 0.0,
        )
        sliced_audio = slice_master_audio_for_h3_plan(master_track, mode_b_plan, mode_b_dir / "audio_slices", "anthem_master")
        seg_32k_paths = [s["path"] for s in sliced_audio["segments"]]

        ref_img_list_b = [args.kf_01_host_hook, args.monkey_avatar] if args.master_locked_dit != "fl2va" else []
        dit_b_map = {
            "ref2va": args.ref2va_dit_path,
            "fl2va": args.fl2va_dit_path,
            "hybrid": args.hybrid_dit_path,
        }
        dit_b_path = dit_b_map[args.master_locked_dit]

        if args.dry_run:
            mode_b_arms_out: Dict[str, Any] = {}
            for arm in arms:
                seg_mp4s = []
                for sidx, seg in enumerate(mode_b_plan["segments"]):
                    smp4 = mode_b_dir / f"mode_b__{arm}__seg{sidx:02d}.mp4"
                    _render_dry_run_anthem_segment(
                        Path(args.kf_01_host_hook),
                        frames=int(seg["frames"]),
                        width=args.width,
                        height=args.height,
                        wav_path=Path(seg_32k_paths[sidx]),
                        out_mp4=smp4,
                        banner_text=f"MODE B MASTER-LOCKED ({arm}) seg{sidx:02d} | YuE2 48kHz",
                    )
                    seg_mp4s.append(smp4)
                final_b = mode_b_dir / f"anthem_mode_b_master_locked_{arm}.mp4"
                stitched = stitch_chained_segments_ffmpeg(
                    seg_mp4s,
                    mode_b_plan,
                    out_mp4=final_b,
                    master_audio_48k=Path(sliced_audio["full_48k_wav"]),
                    target_size=(args.final_width, args.final_height) if args.upscale_output else None,
                    x264_threads=args.x264_threads,
                )
                mode_b_arms_out[arm] = stitched
                f_ex = extract_frames_at_indices(
                    final_b,
                    [0, max(0, mode_b_plan["total_frames"] // 2), max(0, mode_b_plan["total_frames"] - 2)],
                    mode_b_dir / "qa_frames",
                    f"mode_b_{arm}",
                )
                for fr in f_ex:
                    qa_tiles.append({"path": fr["path"], "label": f"ModeB Locked ({arm}) f={fr['frame_index']}"})
            mode_b_hw = detect_hardware_info()
        else:
            job_json = mode_b_dir / "mode_b_job.json"
            atomic_write_json(
                job_json,
                {
                    "mode_tag": "mode_b_master_locked",
                    "out_dir": str(mode_b_dir),
                    "cond_dir": proj["cond_dir"],
                    "prompt_id": "anthem_mode_b_master_locked",
                    "arms": arms,
                    "dit_path": str(dit_b_path),
                    "video_vae_path": str(args.video_vae_path),
                    "audio_vae_path": str(args.audio_vae_path),
                    "first_frame_path": str(args.kf_01_host_hook),
                    "ref_image_paths": ref_img_list_b,
                    "ref_audio_paths": [],
                    "segment_master_audio_32k_paths": seg_32k_paths,
                    "plan": mode_b_plan,
                    "seed": args.seed + 500,
                    "width": args.width,
                    "height": args.height,
                    "steps": args.steps,
                    "cfg": args.cfg,
                    "shift_video": args.shift_video,
                    "shift_audio": args.shift_audio,
                    "sampler_name": args.sampler_name,
                    "scheduler_name": args.scheduler_name,
                    "use_sparse_attention": not args.no_sparse_attention,
                    "video_budget": args.video_budget,
                    "early_steps": args.early_steps,
                    "late_steps": args.late_steps,
                    "resume_segments": bool(args.resume_segments),
                },
            )
            proc = subprocess.run(
                [
                    str(args.comfy_python), "-B", str(Path(__file__).resolve()),
                    "--worker", "h3-render",
                    "--worker-job-json", str(job_json),
                    "--comfy-root", str(args.comfy_root),
                    "--repo-root", str(args.repo_root),
                ],
                check=False,
            )
            if proc.returncode != 0:
                raise RuntimeError(f"Mode B Master-Locked worker failed rc={proc.returncode}")
            r_res = json.loads((mode_b_dir / "mode_b_master_locked_render_result.json").read_text(encoding="utf-8"))
            mode_b_hw = r_res["hardware"]
            mode_b_arms_out = {}
            for arm in arms:
                seg_mp4s = [Path(s["mp4_path"]) for s in r_res["arms"][arm]["segments"]]
                final_b = mode_b_dir / f"anthem_mode_b_master_locked_{arm}.mp4"
                stitched = stitch_chained_segments_ffmpeg(
                    seg_mp4s,
                    mode_b_plan,
                    out_mp4=final_b,
                    master_audio_48k=Path(sliced_audio["full_48k_wav"]),
                    target_size=(args.final_width, args.final_height) if args.upscale_output else None,
                    x264_threads=args.x264_threads,
                )
                mode_b_arms_out[arm] = stitched
                f_ex = extract_frames_at_indices(
                    final_b,
                    [0, max(0, mode_b_plan["total_frames"] // 2), max(0, mode_b_plan["total_frames"] - 2)],
                    mode_b_dir / "qa_frames",
                    f"mode_b_{arm}",
                )
                for fr in f_ex:
                    qa_tiles.append({"path": fr["path"], "label": f"ModeB Locked ({arm}) f={fr['frame_index']}"})

        trace.append(
            stage="packetA_fl2va_av" if args.master_locked_dit == "fl2va" else "packetA_ref2va_av",
            inputs={
                "mode": "master-locked",
                "dit_variant": args.master_locked_dit,
                "zero_qwen_loaded": True,
                "bridge_key_sha256": asset_audits["concat4_key"]["sha256"],
                "master_audio_48k_sha256": sliced_audio["full_48k_sha256"],
                "cfg": args.cfg,
                "shift_video": args.shift_video,
                "shift_audio": args.shift_audio,
                "plan": mode_b_plan,
            },
            outputs={"arms": mode_b_arms_out},
            hardware=mode_b_hw,
            parent_step_hashes=[step3_sha, step4["step_sha256"]],
        )
        deliverables["mode_b_master_locked"] = mode_b_arms_out

    # ------------------------------------------------------------------
    # Stage 6: Contact Sheet + Final Trace Verification & Summary Report
    # ------------------------------------------------------------------
    sheet_info = build_contact_sheet(
        qa_tiles,
        out_dir / "qa_contact_sheet_national_anthem.png",
        cols=3,
        tile_w=640,
        tile_h=362,
        header_title=(
            f"#912 OMNI-INKLING SHOWCASE 1: MONKEY FOUNDER — THE STAR-SPANGLED BANNER "
            f"| Packet A concat4 ({EXPECTED_CONCAT4_KEY_SHA256[:8]})"
        ),
    )
    trace.append(
        stage="ffmpeg_mux",
        inputs={"deliverable_modes": sorted(deliverables.keys())},
        outputs={
            "qa_contact_sheet": sheet_info,
            "deliverables": deliverables,
        },
    )
    trace_audit = verify_trace_jsonl(out_dir / "showcase_process_trace.jsonl")
    if not trace_audit["verified"]:
        raise RuntimeError(f"Showcase trace verification failed: {trace_audit['errors']}")

    report = {
        "schema_version": "omni-912-showcase-national-anthem-report/v1",
        "dry_run": bool(args.dry_run),
        "mode": args.mode,
        "zero_qwen_loaded": True,
        "concat4_key_sha256": asset_audits["concat4_key"]["sha256"],
        "owner_timbre_excerpt": ({**selection, **timbre_wavs} if (selection and timbre_wavs) else None),
        "deliverables": deliverables,
        "qa_contact_sheet": sheet_info,
        "trace_audit": trace_audit,
        "elapsed_seconds": round(time.monotonic() - t0, 2),
    }
    rep_path = out_dir / "national_anthem_showcase_report.json"
    rep_sha = atomic_write_json(rep_path, report)
    print(
        json.dumps(
            {
                "status": "PASS",
                "dry_run": bool(args.dry_run),
                "report": str(rep_path),
                "report_sha256": rep_sha,
                "trace_verified": trace_audit["verified"],
                "trace_steps": trace_audit["n_steps"],
                "qa_contact_sheet": sheet_info["path"],
                "elapsed_s": report["elapsed_seconds"],
            },
            indent=2,
        ),
        flush=True,
    )
    return 0


def build_arg_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="#912 Flagship Showcase 1: Monkey Founder National Anthem (FL2VA/Ref2VA + YuE2 + Plato's Key)")
    ap.add_argument("--worker", choices=["none", "yue2-sing", "h3-render"], default="none")
    ap.add_argument("--worker-job-json", type=str, default=None)
    ap.add_argument(
        "--profile",
        choices=list(ANTHEM_PROFILES.keys()),
        default=None,
        help="RTX PRO 6000 preset: rtxp-60s-full (Mode A 33.5s/804f + Mode B full 60.6s/1467f default), rtxp-30s (33.5s/804f both), smoke-14s (14.4s/345f)",
    )
    ap.add_argument("--mode", choices=["ref2va-timbre", "master-locked", "both"], default="both")
    ap.add_argument("--master-locked-dit", choices=["ref2va", "fl2va", "hybrid"], default="ref2va")
    ap.add_argument("--out-dir", type=str, default="/mnt/storage/omni_gate2_runs/phase0_912_spinup/showcase_runs/national_anthem")
    ap.add_argument("--dry-run", action="store_true", help="Zero-GPU end-to-end CPU dry run verifying pins, AV grid, audio slicing, ffmpeg mux, and JSONL hash chain")
    ap.add_argument("--dry-run-duration-s", type=float, default=3.5)
    ap.add_argument("--resume-segments", action="store_true", help="Reuse completed segXX.mp4 + segXX_tail_latents.pt when resuming multi-segment runs")
    ap.add_argument("--allow-unpinned-assets", action="store_true")
    ap.add_argument("--include-b0-control", action="store_true", help="Also render matched b0_shuffled_single_tap failure control")
    ap.add_argument("--include-cta-ref", action="store_true", help="Also include kf_06_host_cta.png as <Picture 3> in Mode A")
    ap.add_argument("--use-yue2-guide-in-mode-a", action="store_true", help="Pass first 12s of YuE2 anthem as <Audio 2> alongside owner timbre <Audio 1>")
    ap.add_argument("--short-verse", action="store_true", help="Use opening 4-line verse instead of full 8-line Verse 1")
    ap.add_argument("--skip-whisper", action="store_true")

    # Owner pinned inputs
    ap.add_argument("--monkey-avatar", default=PINNED_OWNER_ASSETS["monkey_avatar"]["path"])
    ap.add_argument("--kf-01-host-hook", default=PINNED_OWNER_ASSETS["kf_01_host_hook"]["path"])
    ap.add_argument("--kf-06-host-cta", default=PINNED_OWNER_ASSETS["kf_06_host_cta"]["path"])
    ap.add_argument("--owner-voice-wav", default=PINNED_OWNER_ASSETS["owner_manifesto_wav"]["path"])
    ap.add_argument("--owner-voice-whisper", default=PINNED_OWNER_ASSETS["owner_manifesto_whisper"]["path"])
    ap.add_argument("--timbre-min-s", type=float, default=10.0)
    ap.add_argument("--timbre-max-s", type=float, default=12.5)
    ap.add_argument("--prebuilt-yue2-wav", default=None, help="Reuse already-rendered 48 kHz YuE2 Star-Spangled Banner WAV")
    ap.add_argument("--prebuilt-yue2-abc", default=None)
    ap.add_argument("--prebuilt-conds-dir", default=None, help="Reuse pre-projected <1 MiB Packet A concat4 .pt conds (zero Inkling/exporter needed on G4-4)")
    ap.add_argument("--prebuilt-direction-json", default=None, help="Reuse already-authored inkling_authored_direction.json (zero ingine-serve needed on G4-4)")
    ap.add_argument("--project-conds-only", action="store_true", help="Stop right after projecting Packet A .pt conds + conds_manifest.json for handoff to G4-4")
    ap.add_argument("--master-audio-override", default=None, help="Optional custom 48 kHz master audio (e.g. timbre-converted vocal mix) for Mode B")

    # Fleet-configurable model & code paths (Devbox PRO 6000 / Colab G4-4 / Vast)
    ap.add_argument("--repo-root", default=DEFAULT_PATHS["repo_root"])
    ap.add_argument("--comfy-root", default=DEFAULT_PATHS["comfy_root"])
    ap.add_argument("--comfy-python", default=DEFAULT_PATHS["comfy_python"])
    ap.add_argument("--keys-dir", default=DEFAULT_PATHS["keys_dir"])
    ap.add_argument("--inkling-dir", default=DEFAULT_PATHS["inkling_dir"])
    ap.add_argument("--exporter-bin", default=DEFAULT_PATHS["exporter_bin"])
    ap.add_argument("--ingine-serve-bin", default=DEFAULT_PATHS["ingine_serve_bin"])
    ap.add_argument("--inkling-url", default=None, help="Optional already-running OpenAI-compatible ingine-serve base URL")
    ap.add_argument("--persona-spec", default=None, help="Optional visit-indexed Inkling persona spec JSON (visits 41..47)")
    ap.add_argument("--fl2va-dit-path", default=DEFAULT_PATHS["fl2va_dit_path"])
    ap.add_argument("--ref2va-dit-path", default=DEFAULT_PATHS["ref2va_dit_path"])
    ap.add_argument("--hybrid-dit-path", default=DEFAULT_PATHS["hybrid_dit_path"])
    ap.add_argument("--video-vae-path", default=DEFAULT_PATHS["video_vae_path"])
    ap.add_argument("--audio-vae-path", default=DEFAULT_PATHS["audio_vae_path"])
    ap.add_argument("--whisper-dir", default=DEFAULT_PATHS["whisper_dir"])
    ap.add_argument("--qwen-tts-deps", default=DEFAULT_PATHS["qwen_tts_deps"])
    ap.add_argument("--yue2-deps", default=DEFAULT_PATHS["yue2_deps"])
    ap.add_argument("--yue2-model-dir", default=DEFAULT_PATHS["yue2_model_dir"])
    ap.add_argument("--yue2-vae-dir", default=DEFAULT_PATHS["yue2_vae_dir"])
    ap.add_argument("--yue2-budget-gib", type=int, default=24)
    ap.add_argument("--yue2-ode-steps", type=int, default=24)

    # MiniMax-H3 sampling parameters (agreed defaults: cfg=1.0, shift_video=12.0, shift_audio=3.0)
    ap.add_argument("--seed", type=int, default=20261127)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--final-width", type=int, default=1920)
    ap.add_argument("--final-height", type=int, default=1088)
    ap.add_argument("--upscale-output", action="store_true")
    ap.add_argument("--steps", type=int, default=16)
    ap.add_argument("--cfg", type=float, default=1.0)
    ap.add_argument("--shift-video", type=float, default=12.0)
    ap.add_argument("--shift-audio", type=float, default=3.0)
    ap.add_argument("--sampler-name", default="res_multistep")
    ap.add_argument("--scheduler-name", default="simple")
    ap.add_argument("--no-sparse-attention", action="store_true")
    ap.add_argument("--video-budget", type=float, default=0.30)
    ap.add_argument("--early-steps", type=int, default=4)
    ap.add_argument("--late-steps", type=int, default=2)
    ap.add_argument("--mode-a-duration-s", type=float, default=None, help="Mode A Ref2VA target seconds (default 33.5s / 804f coherent >=30s RTX PRO 6000)")
    ap.add_argument("--max-duration-s", type=float, default=None, help="Optional Mode B cap in seconds (default None = full YuE2 track, 60.64s -> 1467f)")
    ap.add_argument("--max-seg-frames", type=int, default=345)
    ap.add_argument("--x264-threads", type=int, default=4)
    return ap


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = build_arg_parser()
    args = ap.parse_args(argv)
    if args.worker == "yue2-sing":
        worker_yue2_sing(args)
        return 0
    if args.worker == "h3-render":
        worker_h3_anthem_render(args)
        return 0
    return run_coordinator(args)


if __name__ == "__main__":
    raise SystemExit(main())

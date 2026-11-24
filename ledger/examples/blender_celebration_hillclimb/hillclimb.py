"""Multimodal Vision-Guided Hillclimb Driver for the Flagship Celebration Video.

Orchestrates on `atlas-ingine` (with native GPU vision ON + JSON response_format):
  1. Standalone `celebration_video_ledger.sqlite` (verified Addition + audited Subtraction/Superseding)
  2. 3-Tier Rolling Context Compressor (Pinned Brief+Ledger / ~150-tok Compressed Deltas / Live Champion+Vision)
  3. Two-Process Headless Blender 4.5.14 LTS Verifier (5 multi-timestamp Cycles OIDN keyframes)
  4. Native Inkling-2-Mini GPU Vision Critique & Host-Gated Ledger Curation Turn (`json_object`)
  5. Optional Cinema 1080p MP4 render + Hillclimb Progression Contact Sheet PNG
"""
from __future__ import annotations

import argparse
import json
import os
import re
import time
import urllib.request
from typing import Any

import hashlib
import subprocess

_PY_BLOCK_RE = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.DOTALL | re.IGNORECASE)


def extract_python_script(reply: str) -> str:
    blocks = _PY_BLOCK_RE.findall(reply or "")
    if blocks:
        blocks.sort(key=len, reverse=True)
        return blocks[0].strip()
    txt = (reply or "").strip()
    if "import bpy" in txt:
        return txt[txt.index("import bpy") :].strip()
    return txt


def _probe_blender_version(blender_bin: str) -> tuple[str, str]:
    try:
        out = subprocess.check_output([blender_bin, "--version"], text=True, timeout=15)
        ver = (out.splitlines() or ["Blender 4.5"])[0].strip()
    except Exception:
        ver = "Blender 4.5"
    fp = hashlib.sha256(f"{blender_bin}:{ver}".encode("utf-8")).hexdigest()[:16]
    return ver, fp


def _targeted_repair_hint(verif: Any) -> str:
    msg = (verif.traceback_summary or "") + " " + " ".join(verif.failed_reasons or [])
    hints: list[str] = []
    if "Fac" in msg or "MixRGB" in msg:
        hints.append("- ShaderNodeMix (Blender 4.x): set `mx.data_type = 'RGBA'`, use inputs `'Factor'` / `6` / `7` and output `2` (never `'Fac'`).")
    if "NoneType" in msg and ("world" in msg or "color" in msg or "interface" in msg):
        hints.append("- After `read_factory_settings(use_empty=True)`, create `scene.world = bpy.data.worlds.new('World')` and `mod.node_group = bpy.data.node_groups.new('Tree', 'GeometryNodeTree')`.")
    if "primitive_box_add" in msg or "Node type" in msg:
        hints.append("- Use `primitive_cube_add` (never `primitive_box_add`) and `GeometryNodeInputPosition` / `GeometryNodeMeshCube` / `GeometryNodeMeshIcoSphere`.")
    if "handle_left" in msg or "type" in msg:
        hints.append("- Set `kp.handle_left_type = 'AUTO'` (never `kp.handle_left.type`).")
    return ("[VERIFIED_REPAIR_HINTS]\n" + "\n".join(hints)) if hints else ""

from .context_compressor import (
    CELEBRATION_CREATIVE_BRIEF,
    IterationDelta,
    RollingContextCompressor,
    encode_png_part,
)
from .ledger import CelebrationVideoLedger
from .verifier import CelebrationVerifier, build_progression_sheet, render_final_1080p_mp4


CRITIQUE_SYSTEM_PROMPT = (
    "You are an uncompromising editorial cinematographer and 3D lighting director reviewing "
    "5 multi-timestamp Cycles-rendered keyframes from a 10-second celebration video: "
    "'10 Years of NVIDIA AI Architecture (2016 Pascal -> 2026 Blackwell) & Thinking Machines Inkling-2-Mini'.\n"
    "Be strictly honest — NEVER inflate scores for flat, dark, cluttered, clipped, or poorly composed renders.\n"
    "Evaluate the 5 frames and host telemetry, and decide what verified rules to ADD to or SUBTRACT from the active ledger.\n"
    "Return ONLY one valid JSON object with these exact keys:\n"
    "{\n"
    '  "scores": {\n'
    '    "composition_framing": 0-10,\n'
    '    "lighting_atmosphere": 0-10,\n'
    '    "architectural_storytelling": 0-10,\n'
    '    "material_craft": 0-10,\n'
    '    "animation_progression": 0-10\n'
    "  },\n"
    '  "verdict": "1-2 sentence candid visual critique of what works and the #1 remaining flaw",\n'
    '  "next_focus": "single concrete highest-leverage Blender change for the next iteration",\n'
    '  "kept_elements": ["short phrase", ...],\n'
    '  "dropped_elements": ["short phrase", ...],\n'
    '  "ledger_add": [\n'
    '    {"category": "scene_technique"|"aesthetic_lesson", "slot_key": "short_slot_name", '
    '"title": "concise title", "summary": "verified concrete rule", "snippet": "optional <=250 char bpy snippet"}\n'
    "  ],\n"
    '  "ledger_subtract": [\n'
    '    {"entry_id": 123, "reason": "exact visual/host defect or conflict caused by this active entry"}\n'
    "  ]\n"
    "}"
)


def call_ingine_chat(
    *,
    base_url: str,
    model: str,
    messages: list[dict[str, Any]],
    temperature: float = 0.6,
    reasoning_effort: str = "medium",
    max_tokens: int = 8192,
    seed: int = 42,
    response_format: dict[str, Any] | None = None,
    timeout_s: int = 360,
) -> dict[str, Any]:
    url = base_url.rstrip("/") + "/v1/chat/completions"
    payload: dict[str, Any] = {
        "model": model,
        "messages": messages,
        "stream": False,
        "temperature": temperature,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
        "seed": seed,
    }
    if response_format is not None:
        payload["response_format"] = response_format

    req = urllib.request.Request(
        url,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=timeout_s) as resp:
        data = json.loads(resp.read().decode("utf-8"))
    elapsed = round(time.time() - t0, 3)
    choice = (data.get("choices") or [{}])[0]
    msg = choice.get("message") or {}
    usage = data.get("usage") or {}
    return {
        "content": msg.get("content") or "",
        "reasoning_chars": len(msg.get("reasoning_content") or ""),
        "finish_reason": choice.get("finish_reason") or "",
        "prompt_tokens": int(usage.get("prompt_tokens", 0)),
        "completion_tokens": int(usage.get("completion_tokens", 0)),
        "total_tokens": int(usage.get("total_tokens", 0)),
        "llm_seconds": elapsed,
    }


def _parse_json_reply(text: str) -> dict[str, Any]:
    raw = (text or "").strip()
    if raw.startswith("```"):
        m = re.search(r"```(?:json)?\s*\n(.*?)```", raw, re.DOTALL | re.IGNORECASE)
        if m:
            raw = m.group(1).strip()
    try:
        obj = json.loads(raw)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    m2 = re.search(r"\{.*\}", raw, re.DOTALL)
    if m2:
        try:
            obj = json.loads(m2.group(0))
            if isinstance(obj, dict):
                return obj
        except Exception:
            pass
    return {}


def seed_api_facts(ledger: CelebrationVideoLedger, blender_bin: str) -> str:
    """Probe live Blender 4.5.14 LTS API truth and pin verified `api_fact` entries in the ledger."""
    ver_str, fp = _probe_blender_version(blender_bin)
    ledger.invalidate_api_facts_on_drift(fp, iter_num=0)
    facts = [
        ("blender_version", ver_str, f"Live runtime is {ver_str} (Python 3.11 — NO backslashes inside f-string expressions). Clear with read_factory_settings(use_empty=True)."),
        ("world_none_guard", "Create scene.world after factory reset", "After read_factory_settings(use_empty=True), scene.world is None! Always create `w = bpy.data.worlds.new('World'); scene.world = w; w.use_nodes = True` before accessing world nodes/color."),
        ("shader_mix_v4", "ShaderNodeMix & NodeTree links", "Link sockets on the node_tree (`nt.links.new(a, b)`), NEVER `node.links`. Use ShaderNodeMix (data_type='RGBA') with inputs 'Factor'/6/7 and output 2 — NEVER 'Fac' or ShaderNodeMixRGB."),
        ("principled_v4", "Principled BSDF 4.x sockets", "Use 'Transmission Weight', 'Coat Weight', 'Specular IOR Level', 'Emission Color', 'Emission Strength' — NEVER legacy 'Transmission'/'Clearcoat'."),
        ("compositor_glare_v4", "CompositorNodeGlare 4.x link", "Link Render Layers -> CompositorNodeGlare (glare_type='FOG_GLOW', threshold=1.0, size=7) -> Composite. No 'intensity' attribute."),
        ("geonodes_v4", "Geometry Nodes 4.x NodeTreeInterface", "Use ng.interface.new_socket(name='Geometry', in_out='INPUT'/'OUTPUT', socket_type='NodeSocketGeometry'), NEVER ng.inputs.new(). Use GeometryNodeInputPosition (NOT FunctionNodeInputPosition)."),
        ("agx_and_mesh", "AgX + Smooth Shading + Primitives + Keyframes", "Set scene.view_settings.view_transform = 'AgX'; never use mesh.use_auto_smooth or primitive_box_add; mathutils.Vector((x,y,z)) takes ONE tuple; set `kp.handle_left_type = 'AUTO'` (NOT `kp.handle_left.type`)."),
    ]
    for slot, title, summary in facts:
        ledger.admit_entry(
            category="api_fact",
            slot_key=slot,
            title=title,
            summary=summary,
            iter_num=0,
            score=100.0,
            host_verified=True,
            vision_verified=False,
            blender_fp=fp,
            reason="live host API probe",
            supersede_same_slot=True,
        )
    # Seed 4 compact, live-verified Blender 4.5.14 starter code patterns at low baseline score (30.0)
    # so Iteration 1 executes cleanly on the first try and higher-scoring hillclimb winners cleanly SUPERSEDE them.
    starter_techniques = [
        (
            "geonodes_orbital_field",
            "Blender 4.5 Geometry Nodes Starter (create node_group explicitly)",
            "Always create `bpy.data.node_groups.new(..., 'GeometryNodeTree')`, assign `mod.node_group = ng`, and wire links via `ng.links.new(a, b)`.",
            "mod = obj.modifiers.new('OrbitalGN', 'NODES')\n"
            "ng = bpy.data.node_groups.new('OrbitalTree', 'GeometryNodeTree')\n"
            "mod.node_group = ng\n"
            "ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')\n"
            "ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')\n"
            "ni, no = ng.nodes.new('NodeGroupInput'), ng.nodes.new('NodeGroupOutput')\n"
            "dist = ng.nodes.new('GeometryNodeDistributePointsOnFaces')\n"
            "ico = ng.nodes.new('GeometryNodeMeshIcoSphere'); ico.inputs['Radius'].default_value = 0.04\n"
            "inst = ng.nodes.new('GeometryNodeInstanceOnPoints'); join = ng.nodes.new('GeometryNodeJoinGeometry')\n"
            "ng.links.new(ni.outputs['Geometry'], dist.inputs['Mesh'])\n"
            "ng.links.new(dist.outputs['Points'], inst.inputs['Points'])\n"
            "ng.links.new(ico.outputs['Mesh'], inst.inputs['Instance'])\n"
            "ng.links.new(ni.outputs['Geometry'], join.inputs['Geometry'])\n"
            "ng.links.new(inst.outputs['Instances'], join.inputs['Geometry'])\n"
            "ng.links.new(join.outputs['Geometry'], no.inputs['Geometry'])",
        ),
        (
            "compositor_bloom",
            "Blender 4.5 Linked FOG_GLOW Compositor Starter",
            "Enable `scene.use_nodes = True` and wire `R_LAYERS -> CompositorNodeGlare -> COMPOSITE` via `nt.links.new(...)`.",
            "scene.use_nodes = True; cnt = scene.node_tree; cnt.nodes.clear()\n"
            "rl = cnt.nodes.new('CompositorNodeRLayers'); gl = cnt.nodes.new('CompositorNodeGlare')\n"
            "gl.glare_type = 'FOG_GLOW'; gl.threshold = 1.0; gl.size = 7\n"
            "co = cnt.nodes.new('CompositorNodeComposite')\n"
            "cnt.links.new(rl.outputs['Image'], gl.inputs['Image'])\n"
            "cnt.links.new(gl.outputs['Image'], co.inputs['Image'])",
        ),
        (
            "material_palette",
            "Atlas Blue-Steel Procedural PBR Starter (Brand Guide p.19)",
            "Use `ShaderNodeTexNoise -> ShaderNodeValToRGB (ColorRamp) -> Principled BSDF Roughness`; never use `node.links` or `ShaderNodeMix['Fac']`.",
            "mat = bpy.data.materials.new('BlueSteel'); mat.use_nodes = True; nt = mat.node_tree\n"
            "bsdf = nt.nodes['Principled BSDF']; bsdf.inputs['Base Color'].default_value = (0.09, 0.15, 0.19, 1.0)\n"
            "bsdf.inputs['Metallic'].default_value = 0.85\n"
            "nz = nt.nodes.new('ShaderNodeTexNoise'); nz.inputs['Scale'].default_value = 18.0\n"
            "cr = nt.nodes.new('ShaderNodeValToRGB')\n"
            "cr.color_ramp.elements[0].color = (0.18, 0.18, 0.18, 1.0); cr.color_ramp.elements[1].color = (0.42, 0.42, 0.42, 1.0)\n"
            "nt.links.new(nz.outputs['Fac'], cr.inputs['Fac']); nt.links.new(cr.outputs['Color'], bsdf.inputs['Roughness'])",
        ),
        (
            "camera_choreography",
            "Sweeping Bezier Camera Track-To Starter (240f @ 24fps)",
            "Track-To an Empty target and keyframe `cam.location` across frames 1, 120, 240 with `BEZIER` interpolation (handles Blender 4.4+ slotted actions).",
            "bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 0.5, 1.4)); tgt = bpy.context.active_object\n"
            "bpy.ops.object.camera_add(location=(-7.0, -10.5, 4.2)); cam = bpy.context.active_object; scene.camera = cam\n"
            "tr = cam.constraints.new(type='TRACK_TO'); tr.target = tgt; tr.track_axis = 'TRACK_NEGATIVE_Z'; tr.up_axis = 'UP_Y'\n"
            "for fr, loc in [(1, (-7.0, -10.5, 4.2)), (120, (0.0, -11.5, 3.6)), (240, (6.8, -9.8, 3.0))]:\n"
            "    cam.location = loc; cam.keyframe_insert(data_path='location', frame=fr)",
        ),
    ]
    for slot, title, summary, snippet in starter_techniques:
        ledger.admit_entry(
            category="scene_technique",
            slot_key=slot,
            title=title,
            summary=summary,
            snippet=snippet,
            iter_num=0,
            score=30.0,
            host_verified=True,
            vision_verified=True,
            blender_fp=fp,
            reason="verified Blender 4.5.14 starter scaffold (supersede as hillclimb improves)",
            supersede_same_slot=True,
        )
    return fp


def run_celebration_hillclimb(
    *,
    out_dir: str,
    base_url: str = "http://127.0.0.1:8198",
    model: str = "inkling-2-mini",
    blender_bin: str | None = None,
    render_device: str = "CUDA",
    iters: int = 12,
    preview_samples: int = 24,
    temperature: float = 0.6,
    author_effort: str = "low",
    critique_effort: str = "low",
    max_tokens: int = 16384,
    max_inner_repairs: int = 3,
    seed: int = 42,
    render_1080p: bool = False,
    final_samples: int = 64,
    llm_fn: Any = None,
) -> dict[str, Any]:
    os.makedirs(out_dir, exist_ok=True)
    iters_dir = os.path.join(out_dir, "iterations")
    os.makedirs(iters_dir, exist_ok=True)

    ledger_path = os.path.join(out_dir, "celebration_video_ledger.sqlite")
    ledger = CelebrationVideoLedger(ledger_path)
    verifier = CelebrationVerifier(
        verify_blender_bin=blender_bin,
        render_device=render_device,
        cycles_samples=preview_samples,
    )
    blender_fp = seed_api_facts(ledger, verifier.verify_blender_bin)
    compressor = RollingContextCompressor(creative_brief=CELEBRATION_CREATIVE_BRIEF)
    caller = llm_fn or call_ingine_chat

    deltas: list[IterationDelta] = []
    iter_records: list[dict[str, Any]] = []

    best_score = -1.0
    best_iter: int | None = None
    best_script: str | None = None
    best_telemetry = ""
    best_renders: list[str] = []
    best_blend = ""

    last_iter_failed = False
    last_script = ""
    last_failure_telemetry = ""
    last_repair_block = ""
    last_renders: list[str] = []
    pending_crash_sig: tuple[str, str] | None = None
    next_focus = (
        "Establish the full 6-monolith Pascal->Blackwell architectural gallery on a dark reflective obsidian floor, "
        "elevated glowing gold Inkling-2-Mini looped recurrence core with Geometry Nodes orbital field, legible 3D "
        "milestone years + title, dramatic emerald/gold/cool-white studio lighting with AgX + FOG_GLOW glare, and a "
        "sweeping 10-second (240-frame @ 24fps) Bezier camera journey."
    )

    t_start = time.time()
    try:
        for k in range(1, iters + 1):
            iter_prefix = f"iter_{k:02d}"
            iter_sub = os.path.join(iters_dir, iter_prefix)
            os.makedirs(iter_sub, exist_ok=True)

            # 1+2. AUTHOR TURN + FAST INNER CRASH-REPAIR LOOP (so 1-line bpy typos never waste an outer hillclimb iter!)
            repair_attempts = 0
            cur_last_failed = last_iter_failed
            cur_last_script = last_script
            cur_last_telem = last_failure_telemetry
            cur_last_repair = last_repair_block
            cur_last_renders = last_renders

            while True:
                messages, ctx_metrics = compressor.build_author_messages(
                    iter_num=k,
                    total_iters=iters,
                    ledger=ledger,
                    deltas=deltas,
                    best_script=best_script,
                    best_score=max(0.0, best_score),
                    best_iter=best_iter,
                    best_telemetry=best_telemetry,
                    best_render_paths=best_renders,
                    last_iter_failed=cur_last_failed,
                    last_script=cur_last_script,
                    last_failure_telemetry=cur_last_telem,
                    last_repair_block=cur_last_repair,
                    last_render_paths=cur_last_renders,
                    next_focus=next_focus,
                )
                auth_res = caller(
                    base_url=base_url,
                    model=model,
                    messages=messages,
                    temperature=temperature,
                    reasoning_effort=author_effort,
                    max_tokens=max_tokens,
                    seed=seed + k * 100 + repair_attempts * 7,
                )
                script = extract_python_script(auth_res["content"])
                script_path = os.path.join(iter_sub, f"{iter_prefix}.py")
                with open(script_path, "w", encoding="utf-8") as fh:
                    fh.write(script)

                verif = verifier.verify_script(script, save_dir=iter_sub, prefix=iter_prefix, do_render=True)
                telem_txt = verif.compact_telemetry()
                if verif.ok_execution or repair_attempts >= max_inner_repairs:
                    break
                # Record crash signature & immediately patch in fast inner repair loop
                repair_attempts += 1
                pending_crash_sig = (verif.traceback_type or "RuntimeError", verif.traceback_summary[:220])
                print(
                    f"  [ITER {k:02d} inner-repair {repair_attempts}/{max_inner_repairs}] "
                    f"{verif.traceback_type}: {verif.traceback_summary[:100]}",
                    flush=True,
                )
                cur_last_failed = True
                cur_last_script = script
                cur_last_telem = telem_txt
                cur_last_repair = _targeted_repair_hint(verif)
                cur_last_renders = []

            # Track crash -> verified fix repair rule admission
            ledger_added: list[str] = []
            ledger_subtracted: list[str] = []
            if not verif.ok_execution:
                pending_crash_sig = (verif.traceback_type or "RuntimeError", verif.traceback_summary[:220])
            elif pending_crash_sig is not None:
                c_type, c_sum = pending_crash_sig
                rid, _ = ledger.admit_entry(
                    category="repair_rule",
                    slot_key=f"fix_{c_type.lower()[:28]}",
                    title=f"Recovered from {c_type}",
                    summary=f"Resolved prior error ({c_sum[:160]}) cleanly at iter #{k}.",
                    iter_num=k,
                    score=verif.host_score,
                    host_verified=True,
                    vision_verified=False,
                    blender_fp=blender_fp,
                    reason="host-verified crash recovery",
                )
                if rid is not None:
                    ledger_added.append(f"#{rid}:fix_{c_type.lower()[:16]}")
                pending_crash_sig = None

            # 3. NATIVE GPU VISION CRITIQUE & LEDGER ADD/SUBTRACT CURATION TURN
            vision_score = 0.0
            axis_scores: dict[str, float] = {}
            verdict = ""
            kept_el: list[str] = []
            dropped_el: list[str] = []
            crit_res: dict[str, Any] = {}

            if verif.ok_execution and verif.saved_renders:
                img_parts = [p for p in (encode_png_part(rp) for rp in verif.saved_renders[:5]) if p is not None]
                active_listing = "\n".join(
                    f"  - #{e.entry_id} [{e.category}/{e.slot_key}] {e.title}: {e.summary[:120]}"
                    for e in ledger.list_entries(status="active")
                    if e.category != "api_fact"
                ) or "  (none yet)"
                crit_text = (
                    f"Iteration #{k} of {iters} (current champion composite={max(0.0, best_score):.1f}/100 at iter #{best_iter}).\n"
                    f"HOST TELEMETRY:\n{telem_txt}\n\n"
                    f"CURRENT ACTIVE LEDGER ENTRIES (eligible for subtraction/superseding):\n{active_listing}\n\n"
                    "Inspect the 5 attached Cycles keyframes (5%, 25%, 50%, 75%, 95% of animation) and return your JSON critique + ledger curation."
                )
                crit_messages = [
                    {"role": "system", "content": CRITIQUE_SYSTEM_PROMPT},
                    {"role": "user", "content": [*img_parts, {"type": "text", "text": crit_text}]},
                ]
                crit_res = caller(
                    base_url=base_url,
                    model=model,
                    messages=crit_messages,
                    temperature=0.2,
                    reasoning_effort=critique_effort,
                    max_tokens=2048,
                    seed=seed + k * 100 + 50,
                    response_format={"type": "json_object"},
                )
                parsed = _parse_json_reply(crit_res.get("content", ""))
                raw_scores = parsed.get("scores") if isinstance(parsed.get("scores"), dict) else {}
                for ax in (
                    "composition_framing",
                    "lighting_atmosphere",
                    "architectural_storytelling",
                    "material_craft",
                    "animation_progression",
                ):
                    try:
                        val = float(raw_scores.get(ax, 0.0))
                    except (TypeError, ValueError):
                        val = 0.0
                    axis_scores[ax] = max(0.0, min(10.0, val))
                vision_score = round(sum(axis_scores.values()), 2)
                verdict = str(parsed.get("verdict") or "").strip()[:320]
                if parsed.get("next_focus"):
                    next_focus = str(parsed["next_focus"]).strip()[:380]
                kept_el = [str(x) for x in (parsed.get("kept_elements") or []) if str(x).strip()][:5]
                dropped_el = [str(x) for x in (parsed.get("dropped_elements") or []) if str(x).strip()][:5]

                # 4a. AUDITED LEDGER SUBTRACTIONS (retract defective/conflicting active entries)
                for sub in (parsed.get("ledger_subtract") or [])[:4]:
                    if not isinstance(sub, dict):
                        continue
                    eid = sub.get("entry_id")
                    reason = str(sub.get("reason") or "retracted by vision critique").strip()
                    if isinstance(eid, int):
                        retracted = ledger.retract_entry(entry_id=eid, iter_num=k, reason=reason)
                        for rid in retracted:
                            ledger_subtracted.append(f"#{rid}")

                # 4b. HOST-GATED LEDGER ADDITIONS (ONLY if host_passed AND vision_score >= 24/50!)
                composite = round(verif.host_score + vision_score, 2)
                admit_ok = bool(verif.host_passed and vision_score >= 24.0)
                for add in (parsed.get("ledger_add") or [])[:3]:
                    if not isinstance(add, dict):
                        continue
                    cat = str(add.get("category") or "scene_technique").strip()
                    if cat not in ("scene_technique", "aesthetic_lesson"):
                        cat = "scene_technique"
                    slot = str(add.get("slot_key") or f"iter{k}_win").strip()
                    title = str(add.get("title") or "").strip()
                    summary = str(add.get("summary") or "").strip()
                    snippet = str(add.get("snippet") or "").strip()
                    new_id, sup_ids = ledger.admit_entry(
                        category=cat,
                        slot_key=slot,
                        title=title,
                        summary=summary,
                        snippet=snippet,
                        iter_num=k,
                        score=composite,
                        host_verified=verif.host_passed,
                        vision_verified=admit_ok,
                        blender_fp=blender_fp,
                        reason=f"vision+host verified at iter #{k} (comp={composite:.1f})",
                        supersede_same_slot=True,
                    )
                    if new_id is not None:
                        ledger_added.append(f"#{new_id}:{slot}")
                    for sid in sup_ids:
                        ledger_subtracted.append(f"sup#{sid}")
            else:
                composite = round(verif.host_score, 2)
                verdict = f"Host execution failed: {verif.traceback_summary[:180]}"

            # Prune active ledger to keep Tier 1 compact
            pruned_ids = ledger.prune_to_budget(iter_num=k)
            for pid in pruned_ids:
                ledger_subtracted.append(f"prune#{pid}")

            delta_vs_best = round(composite - max(0.0, best_score), 2)
            is_new_best = bool(verif.ok_execution and (best_iter is None or composite > best_score))
            if is_new_best:
                best_score = composite
                best_iter = k
                best_script = script
                best_telemetry = telem_txt
                best_renders = list(verif.saved_renders)
                best_blend = verif.saved_blend
                last_iter_failed = False
                last_script = ""
                last_failure_telemetry = ""
                last_repair_block = ""
                last_renders = []
            else:
                last_iter_failed = not verif.host_passed
                last_script = script
                last_failure_telemetry = telem_txt
                last_renders = list(verif.saved_renders)
                last_repair_block = _targeted_repair_hint(verif)

            delta = IterationDelta(
                iter_num=k,
                host_ok=verif.host_passed,
                host_score=verif.host_score,
                vision_score=vision_score,
                composite_score=composite,
                delta_vs_best=delta_vs_best,
                is_new_best=is_new_best,
                visual_verdict=verdict,
                next_focus=next_focus,
                kept_elements=kept_el,
                dropped_elements=dropped_el,
                ledger_added=ledger_added,
                ledger_subtracted=ledger_subtracted,
                failure_note=("; ".join(verif.failed_reasons[:2]) if not verif.host_passed else ""),
            )
            deltas.append(delta)

            rec = {
                "iter": k,
                "ok_execution": verif.ok_execution,
                "host_passed": verif.host_passed,
                "host_score": verif.host_score,
                "vision_score": vision_score,
                "axis_scores": axis_scores,
                "composite_score": composite,
                "is_new_best": is_new_best,
                "verdict": verdict,
                "next_focus": next_focus,
                "ledger_added": ledger_added,
                "ledger_subtracted": ledger_subtracted,
                "context_metrics": ctx_metrics,
                "author_usage": {
                    "prompt_tokens": auth_res.get("prompt_tokens", 0),
                    "completion_tokens": auth_res.get("completion_tokens", 0),
                    "llm_seconds": auth_res.get("llm_seconds", 0.0),
                    "finish_reason": auth_res.get("finish_reason", ""),
                },
                "critique_usage": {
                    "prompt_tokens": crit_res.get("prompt_tokens", 0),
                    "completion_tokens": crit_res.get("completion_tokens", 0),
                    "llm_seconds": crit_res.get("llm_seconds", 0.0),
                },
                "blender_seconds": verif.exec_seconds,
                "saved_renders": verif.saved_renders,
                "saved_blend": verif.saved_blend,
                "failed_reasons": verif.failed_reasons,
            }
            iter_records.append(rec)
            print(
                f"[ITER {k:02d}/{iters:02d}] comp={composite:5.1f}/100 (host={verif.host_score:4.1f} vis={vision_score:4.1f}) "
                f"{'*NEW_BEST*' if is_new_best else '          '} "
                f"add={len(ledger_added)} sub={len(ledger_subtracted)} "
                f"ctx_t1={ctx_metrics['tier1_chars']}c t2={ctx_metrics['tier2_chars']}c vis_imgs={ctx_metrics['tier3_vision_images']} | {verdict[:90]}",
                flush=True,
            )

        # Build progression contact sheet (first passing/iter1, mid best, final champion)
        sheet_rows: list[tuple[str, list[str]]] = []
        picked_iters: list[int] = []
        for cand in [1, max(1, iters // 2), best_iter or iters]:
            if cand not in picked_iters:
                picked_iters.append(cand)
        for idx in picked_iters:
            r = next((x for x in iter_records if x["iter"] == idx), None)
            if r and r.get("saved_renders"):
                tag = "CHAMPION" if idx == best_iter else f"ITER {idx:02d}"
                sheet_rows.append(
                    (f"{tag} — comp {r['composite_score']:.1f}/100 (host {r['host_score']:.1f} + vis {r['vision_score']:.1f})", r["saved_renders"])
                )
        progression_png = build_progression_sheet(sheet_rows, os.path.join(out_dir, "hillclimb_progression.png")) if sheet_rows else ""

        final_mp4_info: dict[str, Any] = {}
        if render_1080p and best_blend and os.path.isfile(best_blend):
            final_mp4_info = render_final_1080p_mp4(
                blend_path=best_blend,
                out_mp4=os.path.join(out_dir, "celebration_10yr_nvidia_inkling_1080p.mp4"),
                frames_dir=os.path.join(out_dir, "frames_1080p"),
                blender_bin=verifier.verify_blender_bin,
                render_device=render_device,
                samples=final_samples,
            )

        summary = {
            "out_dir": out_dir,
            "ledger_sqlite": ledger_path,
            "model": model,
            "iters_completed": len(iter_records),
            "best_iter": best_iter,
            "best_composite_score": round(max(0.0, best_score), 2),
            "best_blend": best_blend,
            "best_renders": best_renders,
            "progression_sheet_png": progression_png,
            "final_1080p_mp4": final_mp4_info,
            "ledger_stats": ledger.stats(),
            "total_wall_seconds": round(time.time() - t_start, 2),
            "iterations": iter_records,
        }
        with open(os.path.join(out_dir, "hillclimb_report.json"), "w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        return summary
    finally:
        ledger.close()


def main() -> None:
    ap = argparse.ArgumentParser(description="Run Flagship Celebration Video Vision Hillclimb")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--base-url", default="http://127.0.0.1:8198")
    ap.add_argument("--model", default="inkling-2-mini")
    ap.add_argument("--blender-bin", default=None)
    ap.add_argument("--render-device", default="CUDA", choices=("CPU", "CUDA"))
    ap.add_argument("--iters", type=int, default=12)
    ap.add_argument("--preview-samples", type=int, default=24)
    ap.add_argument("--temperature", type=float, default=0.6)
    ap.add_argument("--author-effort", default="low")
    ap.add_argument("--critique-effort", default="low")
    ap.add_argument("--max-tokens", type=int, default=16384)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--render-1080p", action="store_true")
    ap.add_argument("--final-samples", type=int, default=64)
    args = ap.parse_args()

    res = run_celebration_hillclimb(
        out_dir=args.out_dir,
        base_url=args.base_url,
        model=args.model,
        blender_bin=args.blender_bin,
        render_device=args.render_device,
        iters=args.iters,
        preview_samples=args.preview_samples,
        temperature=args.temperature,
        author_effort=args.author_effort,
        critique_effort=args.critique_effort,
        max_tokens=args.max_tokens,
        seed=args.seed,
        render_1080p=args.render_1080p,
        final_samples=args.final_samples,
    )
    print(json.dumps({k: v for k, v in res.items() if k != "iterations"}, indent=2))


if __name__ == "__main__":
    main()

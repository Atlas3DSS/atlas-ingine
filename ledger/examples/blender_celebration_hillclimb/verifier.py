"""Two-Process Deterministic Blender 4.5.14 LTS Verifier & 1080p MP4 Renderer for Celebration Video.

Supports:
  - Fast 5-keyframe Cycles OIDN preview rendering (CPU or CUDA GPU) across 5% / 25% / 50% / 75% / 95%
    of the animation timeline for native `atlas-ingine` GPU vision critique.
  - Quantitative host scoring (0..50 points) across geometry, PBR/procedural shaders, lighting/AgX/glare,
    grounding/framing, denoised exposure/detail, and smooth 5-timestamp Bezier animation motion.
  - Full 1920x1080 24fps Cycles OIDN cinema render + `ffmpeg` bt709 H.264 MP4 encoding +
    side-by-side Hillclimb Progression Contact Sheet PNG.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageFont

import re

_SAVE_EPILOGUE_TEMPLATE = r"""
# === HOST_SAVE_EPILOGUE ===
def __host_save_blend__(__save_cfg__):
    import bpy, json
    bpy.context.view_layer.update()
    bpy.ops.wm.save_as_mainfile(filepath=__save_cfg__["blend_path"])
    with open(__save_cfg__["sentinel_path"], "w", encoding="utf-8") as fh:
        json.dump({"candidate_ok": True, "exec_blender_version": bpy.app.version_string}, fh)

__host_save_blend__(__HOST_SAVE_CFG__)
"""


def resolve_blender_binary(explicit: str | None = None) -> str:
    if explicit and os.path.isfile(explicit):
        return explicit
    env_bin = os.environ.get("BLENDER_BIN", "")
    if env_bin and os.path.isfile(env_bin):
        return env_bin
    for cand in (
        os.path.expanduser("~/.local/opt/blender-4.5.14-linux-x64/blender"),
        shutil.which("blender") or "",
    ):
        if cand and os.path.isfile(cand):
            return cand
    raise FileNotFoundError("Could not locate Blender binary; pass --blender-bin or set BLENDER_BIN.")


def _extract_traceback_summary(stdout: str, stderr: str) -> tuple[str, str, str]:
    blob = (stderr or "") + "\n" + (stdout or "")
    lines = [ln.rstrip() for ln in blob.splitlines() if ln.strip()]
    exc_type, code_line, summary = "", "", ""
    for i, ln in enumerate(lines):
        m = re.match(r"^([A-Za-z_][A-Za-z0-9_.]*(?:Error|Exception)):\s*(.*)$", ln.strip())
        if m:
            exc_type = m.group(1)
            summary = f"{exc_type}: {m.group(2)}"
            if i >= 1 and not lines[i - 1].lstrip().startswith("File "):
                code_line = lines[i - 1].strip()
    if code_line and summary:
        summary = f"{summary} | line: {code_line}"
    return exc_type, code_line, summary


_CELEBRATION_VERIFY_SCRIPT = r"""
import bpy, json, math, os, sys
from mathutils import Vector
from bpy_extras.object_utils import world_to_camera_view

argv = sys.argv
cfg_path = argv[argv.index("--") + 1]
with open(cfg_path, "r", encoding="utf-8") as fh:
    cfg = json.load(fh)

bpy.ops.wm.open_mainfile(filepath=cfg["blend_path"])
scene = bpy.context.scene
depsgraph = bpy.context.evaluated_depsgraph_get()

def _world_bbox(obj):
    eval_obj = obj.evaluated_get(depsgraph)
    corners = [eval_obj.matrix_world @ Vector(c) for c in eval_obj.bound_box]
    xs = [c.x for c in corners]; ys = [c.y for c in corners]; zs = [c.z for c in corners]
    return (min(xs), min(ys), min(zs)), (max(xs), max(ys), max(zs))

def _is_ground_mesh(obj, bmin, bmax):
    sx = bmax[0] - bmin[0]
    sy = bmax[1] - bmin[1]
    sz = bmax[2] - bmin[2]
    return (sx >= 6.0 and sy >= 6.0 and sz <= 0.25 and abs(bmax[2]) <= 0.5)

mesh_objs = [o for o in scene.objects if o.type == "MESH"]
font_objs = [o for o in scene.objects if o.type == "FONT"]
light_objs = [o for o in scene.objects if o.type == "LIGHT"]
cam = scene.camera

ground_objs = []
hero_objs = []
hero_bboxes = {}
total_hero_faces = 0
modifier_types = set()
geonodes_connected = False

for o in mesh_objs:
    bmin, bmax = _world_bbox(o)
    if _is_ground_mesh(o, bmin, bmax):
        ground_objs.append(o)
    else:
        hero_objs.append(o)
        hero_bboxes[o.name] = (bmin, bmax)
        eval_o = o.evaluated_get(depsgraph)
        mesh_eval = eval_o.to_mesh()
        total_hero_faces += len(mesh_eval.polygons)
        eval_o.to_mesh_clear()
        for m in o.modifiers:
            modifier_types.add(m.type)
            if m.type == "NODES" and m.node_group is not None:
                ng = m.node_group
                if len(ng.nodes) >= 3 and len(ng.links) >= 1:
                    geonodes_connected = True

# Material & procedural texture audit
has_principled = False
has_procedural_tex = False
_PROC_TYPES = {
    "TEX_NOISE", "TEX_VORONOI", "TEX_WAVE", "TEX_MUSGRAVE", "TEX_MAGIC",
    "TEX_CHECKER", "TEX_BRICK", "TEX_GRADIENT", "TEX_GABOR", "TEX_WHITE_NOISE",
}
for o in hero_objs + font_objs:
    for slot in o.material_slots:
        mat = slot.material
        if mat and mat.use_nodes and mat.node_tree:
            nt = mat.node_tree
            for n in nt.nodes:
                if n.type == "BSDF_PRINCIPLED":
                    has_principled = True
                if n.type in _PROC_TYPES:
                    if any(lk.from_node == n for lk in nt.links):
                        has_procedural_tex = True

# Lighting, AgX, and linked CompositorNodeGlare audit
active_lights = sum(1 for l in light_objs if getattr(l.data, "energy", 0.0) > 0.0)
view_transform = str(scene.view_settings.view_transform)
compositor_glare_linked = False
if scene.use_nodes and scene.node_tree:
    nt = scene.node_tree
    adj = {}
    for lk in nt.links:
        adj.setdefault(lk.from_node, []).append(lk.to_node)
    rl_nodes = [n for n in nt.nodes if n.type == "R_LAYERS"]
    comp_nodes = {n for n in nt.nodes if n.type == "COMPOSITE"}
    glare_nodes = {n for n in nt.nodes if n.type == "GLARE"}
    def _reaches(starts, targets):
        seen = set(starts); stack = list(starts)
        while stack:
            cur = stack.pop()
            if cur in targets:
                return True
            for nxt in adj.get(cur, []):
                if nxt not in seen:
                    seen.add(nxt); stack.append(nxt)
        return False
    for gn in glare_nodes:
        if _reaches(rl_nodes, {gn}) and _reaches([gn], comp_nodes):
            compositor_glare_linked = True
            break

# Grounding & pairwise deep overlap check
min_hero_z = min((b[0][2] for b in hero_bboxes.values()), default=0.0)
ground_z = max((_world_bbox(g)[1][2] for g in ground_objs), default=0.0)
deep_overlap_pairs = []
names = list(hero_bboxes.keys())
for i in range(len(names)):
    a0, a1 = hero_bboxes[names[i]]
    va = max(1e-6, (a1[0]-a0[0])*(a1[1]-a0[1])*(a1[2]-a0[2]))
    for j in range(i + 1, len(names)):
        b0, b1 = hero_bboxes[names[j]]
        vb = max(1e-6, (b1[0]-b0[0])*(b1[1]-b0[1])*(b1[2]-b0[2]))
        ox = max(0.0, min(a1[0], b1[0]) - max(a0[0], b0[0]))
        oy = max(0.0, min(a1[1], b1[1]) - max(a0[1], b0[1]))
        oz = max(0.0, min(a1[2], b1[2]) - max(a0[2], b0[2]))
        ov = ox * oy * oz
        # Only flag genuine structural interpenetration between comparable-scale hero masses;
        # allow thin decorative inlaid bus-traces/accents (small trim < 5% of host monolith volume).
        if min(va, vb) >= 0.05 * max(va, vb) and min(va, vb) >= 0.015 and (ov / min(va, vb)) > 0.42:
            deep_overlap_pairs.append((names[i], names[j]))

# Animation F-curve & Bezier audit
nonconstant_fcurves = 0
interp_modes = set()
for act in bpy.data.actions:
    fcurves = list(getattr(act, "fcurves", []))
    if not fcurves and hasattr(act, "layers"):
        for lyr in act.layers:
            for strip in getattr(lyr, "strips", []):
                for cb in getattr(strip, "channelbags", []):
                    fcurves.extend(list(getattr(cb, "fcurves", [])))
    for fc in fcurves:
        pts = [kp.co[1] for kp in fc.keyframe_points]
        if len(pts) >= 2 and (max(pts) - min(pts)) > 1e-4:
            nonconstant_fcurves += 1
            for kp in fc.keyframe_points:
                interp_modes.add(str(kp.interpolation))

f_start = int(scene.frame_start)
f_end = int(scene.frame_end)
fps = float(scene.render.fps) / max(1e-6, float(scene.render.fps_base))
dur_sec = max(0.0, (f_end - f_start + 1) / max(1e-6, fps))

span = max(1, f_end - f_start)
sample_frames = sorted(set([
    f_start + int(round(span * 0.05)),
    f_start + int(round(span * 0.25)),
    f_start + int(round(span * 0.50)),
    f_start + int(round(span * 0.75)),
    f_start + int(round(span * 0.95)),
]))

# Configure deterministic Cycles render (CPU or CUDA GPU)
scene.render.engine = "CYCLES"
req_dev = str(cfg.get("render_device", "CPU")).upper()
used_dev = "CPU"
if req_dev == "CUDA":
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        any_gpu = False
        for d in prefs.devices:
            if d.type == "CUDA":
                d.use = True
                any_gpu = True
            else:
                d.use = False
        if any_gpu:
            scene.cycles.device = "GPU"
            used_dev = "CUDA"
        else:
            scene.cycles.device = "CPU"
    except Exception:
        scene.cycles.device = "CPU"
else:
    scene.cycles.device = "CPU"

scene.cycles.samples = int(cfg.get("cycles_samples", 24))
scene.cycles.use_denoising = bool(cfg.get("use_denoising", True))
if hasattr(scene.cycles, "denoiser"):
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
scene.cycles.seed = 42
scene.render.resolution_x = int(cfg.get("res_x", 640))
scene.render.resolution_y = int(cfg.get("res_y", 360))
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"

cam_positions = []
hero_centroids = []
frame_coverages = []
frame_in_view = []
rendered_files = []

for fr in sample_frames:
    scene.frame_set(fr)
    dg = bpy.context.evaluated_depsgraph_get()
    if cam is not None:
        cam_eval = cam.evaluated_get(dg)
        cam_positions.append(tuple(cam_eval.matrix_world.translation))
        # Sample evaluated vertices of hero meshes for framing
        xs, ys = [], []
        in_cnt = 0; tot_cnt = 0
        for o in hero_objs[:24]:
            eo = o.evaluated_get(dg)
            me = eo.to_mesh()
            verts = me.vertices
            step = max(1, len(verts) // 128)
            for idx in range(0, len(verts), step):
                wco = eo.matrix_world @ verts[idx].co
                p = world_to_camera_view(scene, cam_eval, wco)
                tot_cnt += 1
                if p.z > 0.0 and 0.0 <= p.x <= 1.0 and 0.0 <= p.y <= 1.0:
                    in_cnt += 1
                if p.z > 0.0:
                    xs.append(min(1.0, max(0.0, p.x)))
                    ys.append(min(1.0, max(0.0, p.y)))
            eo.to_mesh_clear()
        if xs and ys:
            cov = (max(xs) - min(xs)) * (max(ys) - min(ys)) * 100.0
        else:
            cov = 0.0
        frame_coverages.append(round(cov, 2))
        frame_in_view.append(round(in_cnt / max(1, tot_cnt), 4))
    if hero_objs:
        c = Vector((0.0, 0.0, 0.0))
        for o in hero_objs:
            c += o.evaluated_get(dg).matrix_world.translation
        c /= len(hero_objs)
        hero_centroids.append(tuple(c))

    if cfg.get("do_render", True) and cam is not None:
        out_png = os.path.join(cfg["render_dir"], f"frame_{fr:04d}.png")
        scene.render.filepath = out_png
        bpy.ops.render.render(write_still=True)
        if os.path.isfile(out_png):
            rendered_files.append({"frame": fr, "path": out_png})

def _path_len(pts):
    tot = 0.0
    for a, b in zip(pts, pts[1:]):
        tot += math.sqrt(sum((x - y) ** 2 for x, y in zip(a, b)))
    return round(tot, 4)

report = {
    "verify_blender_version": bpy.app.version_string,
    "render_device_used": used_dev,
    "ground_mesh_count": len(ground_objs),
    "hero_mesh_count": len(hero_objs),
    "font_object_count": len(font_objs),
    "total_hero_faces": int(total_hero_faces),
    "modifier_types": sorted(modifier_types),
    "geonodes_connected": bool(geonodes_connected),
    "has_principled_bsdf": bool(has_principled),
    "has_procedural_texture": bool(has_procedural_tex),
    "active_lights": int(active_lights),
    "view_transform": view_transform,
    "compositor_glare_linked": bool(compositor_glare_linked),
    "has_camera": cam is not None,
    "min_hero_z": round(float(min_hero_z), 4),
    "ground_z": round(float(ground_z), 4),
    "deep_overlap_count": len(deep_overlap_pairs),
    "deep_overlap_pairs": deep_overlap_pairs[:8],
    "nonconstant_fcurves": int(nonconstant_fcurves),
    "interp_modes": sorted(interp_modes),
    "frame_start": f_start,
    "frame_end": f_end,
    "fps": round(fps, 3),
    "duration_sec": round(dur_sec, 3),
    "sample_frames": sample_frames,
    "cam_motion_m": _path_len(cam_positions),
    "hero_motion_m": _path_len(hero_centroids),
    "frame_coverages_pct": frame_coverages,
    "frame_in_view_ratios": frame_in_view,
    "rendered_files": rendered_files,
}
with open(cfg["report_path"], "w", encoding="utf-8") as fh:
    json.dump(report, fh, indent=2)
"""


def _box_blur_3x3(arr: np.ndarray) -> np.ndarray:
    p = np.pad(arr, ((1, 1), (1, 1)), mode="edge")
    return (
        p[:-2, :-2] + p[:-2, 1:-1] + p[:-2, 2:]
        + p[1:-1, :-2] + p[1:-1, 1:-1] + p[1:-1, 2:]
        + p[2:, :-2] + p[2:, 1:-1] + p[2:, 2:]
    ) / 9.0


def analyze_keyframes(rendered_files: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute denoised luminance, contrast, edge energy, and per-pair frame motion diffs."""
    lumas: list[np.ndarray] = []
    per_frame: list[dict[str, float]] = []
    for item in rendered_files:
        p = item["path"]
        if not os.path.isfile(p):
            continue
        im = Image.open(p).convert("RGB")
        arr = np.asarray(im, dtype=np.float32) / 255.0
        luma = 0.2126 * arr[:, :, 0] + 0.7152 * arr[:, :, 1] + 0.0722 * arr[:, :, 2]
        blur = _box_blur_3x3(luma)
        gx = np.abs(np.diff(blur, axis=1)).mean()
        gy = np.abs(np.diff(blur, axis=0)).mean()
        edge = float(0.5 * (gx + gy))
        per_frame.append(
            {
                "frame": int(item["frame"]),
                "mean_luma": round(float(luma.mean()), 4),
                "std_luma": round(float(luma.std()), 4),
                "pct_clipped_black": round(float((luma < 0.015).mean() * 100.0), 2),
                "pct_clipped_white": round(float((luma > 0.985).mean() * 100.0), 2),
                "edge_energy": round(edge, 5),
            }
        )
        lumas.append(blur)

    pair_diffs: list[float] = []
    for a, b in zip(lumas, lumas[1:]):
        pair_diffs.append(round(float(np.abs(a - b).mean()), 5))

    if not per_frame:
        return {"n_frames": 0, "per_frame": [], "pair_diffs": [], "mean_luma": 0.0, "std_luma": 0.0, "edge_energy": 0.0, "frame_diff_mean": 0.0, "frame_diff_min": 0.0}

    return {
        "n_frames": len(per_frame),
        "per_frame": per_frame,
        "pair_diffs": pair_diffs,
        "mean_luma": round(float(np.mean([f["mean_luma"] for f in per_frame])), 4),
        "std_luma": round(float(np.mean([f["std_luma"] for f in per_frame])), 4),
        "pct_clipped_black": round(float(np.mean([f["pct_clipped_black"] for f in per_frame])), 2),
        "pct_clipped_white": round(float(np.mean([f["pct_clipped_white"] for f in per_frame])), 2),
        "edge_energy": round(float(np.mean([f["edge_energy"] for f in per_frame])), 5),
        "frame_diff_mean": round(float(np.mean(pair_diffs)), 5) if pair_diffs else 0.0,
        "frame_diff_min": round(float(np.min(pair_diffs)), 5) if pair_diffs else 0.0,
    }


@dataclass
class CelebrationVerification:
    ok_execution: bool
    host_passed: bool
    host_score: float              # 0..50 quantitative deterministic score
    exec_seconds: float
    checks: dict[str, bool] = field(default_factory=dict)
    failed_reasons: list[str] = field(default_factory=list)
    traceback_type: str = ""
    traceback_line: str = ""
    traceback_summary: str = ""
    scene_report: dict[str, Any] = field(default_factory=dict)
    pixel_stats: dict[str, Any] = field(default_factory=dict)
    saved_renders: list[str] = field(default_factory=list)
    saved_blend: str = ""


    def compact_telemetry(self) -> str:
        if not self.ok_execution:
            return (
                f"EXEC=FAILED ({self.exec_seconds:.1f}s) | {self.traceback_type}: {self.traceback_summary}"
                + (f" | line: {self.traceback_line}" if self.traceback_line else "")
            )
        r = self.scene_report
        p = self.pixel_stats
        covs = r.get("frame_coverages_pct") or []
        invs = r.get("frame_in_view_ratios") or []
        failed = [k for k, v in self.checks.items() if not v]
        lines = [
            f"HOST_SCORE={self.host_score:.1f}/50 | HOST_PASSED={self.host_passed} | exec={self.exec_seconds:.1f}s ({r.get('render_device_used','CPU')})",
            f"GEOMETRY: heroes={r.get('hero_mesh_count',0)} fonts={r.get('font_object_count',0)} faces={r.get('total_hero_faces',0)} "
            f"ground={r.get('ground_mesh_count',0)} mods={r.get('modifier_types',[])} geonodes_ok={r.get('geonodes_connected')} overlaps={r.get('deep_overlap_count',0)}",
            f"LOOKDEV: principled={r.get('has_principled_bsdf')} proc_tex={r.get('has_procedural_texture')} "
            f"lights={r.get('active_lights',0)} view={r.get('view_transform')} glare_linked={r.get('compositor_glare_linked')}",
            f"ANIMATION: frames={r.get('frame_start')}..{r.get('frame_end')} ({r.get('duration_sec')}s @ {r.get('fps')}fps) "
            f"fcurves={r.get('nonconstant_fcurves',0)} interp={r.get('interp_modes',[])} "
            f"cam_motion={r.get('cam_motion_m',0)}m hero_motion={r.get('hero_motion_m',0)}m pair_diffs={p.get('pair_diffs',[])}",
            f"FRAMING_5T: cov_pct={covs} in_view={invs}",
            f"PIXELS: mean_luma={p.get('mean_luma',0)} std_luma={p.get('std_luma',0)} edge={p.get('edge_energy',0)} "
            f"clip_blk={p.get('pct_clipped_black',0)}% clip_wht={p.get('pct_clipped_white',0)}%",
        ]
        if failed:
            lines.append("FAILED_CHECKS: " + "; ".join(self.failed_reasons[:8]))
        return "\n".join(lines)


def score_celebration_host(report: dict[str, Any], px: dict[str, Any]) -> tuple[dict[str, bool], list[str], float]:
    """Evaluate quantitative host checks and return (checks, reasons, host_score_out_of_50)."""
    checks: dict[str, bool] = {}
    reasons: list[str] = []
    score = 0.0

    def _chk(name: str, ok: bool, pts: float, fail_msg: str) -> None:
        nonlocal score
        checks[name] = bool(ok)
        if ok:
            score += pts
        else:
            reasons.append(f"{name}: {fail_msg}")

    _chk("zero_traceback", True, 5.0, "")
    _chk("ground_stage", int(report.get("ground_mesh_count", 0)) >= 1, 3.0, "no flat studio floor >= 6m x 6m near z=0")
    _chk("hero_mesh_count", int(report.get("hero_mesh_count", 0)) >= 6, 4.0, f"heroes={report.get('hero_mesh_count',0)} < 6")
    _chk("hero_face_richness", int(report.get("total_hero_faces", 0)) >= 1200, 3.0, f"faces={report.get('total_hero_faces',0)} < 1200")
    mods = set(report.get("modifier_types") or [])
    _chk("bevel_modifier", "BEVEL" in mods, 2.5, f"mods={sorted(mods)} missing BEVEL")
    _chk("geonodes_connected", bool(report.get("geonodes_connected")), 3.5, "no active connected Geometry Nodes ('NODES') tree")
    _chk("no_deep_overlap", int(report.get("deep_overlap_count", 0)) <= 1, 3.0, f"deep_overlaps={report.get('deep_overlap_count',0)} {report.get('deep_overlap_pairs',[])}")
    _chk("principled_and_procedural", bool(report.get("has_principled_bsdf")) and bool(report.get("has_procedural_texture")), 3.0, "missing Principled BSDF or linked procedural texture")
    _chk("lighting_and_agx", int(report.get("active_lights", 0)) >= 3 and "AgX" in str(report.get("view_transform", "")), 3.0, f"lights={report.get('active_lights',0)}, view={report.get('view_transform')}")
    _chk("compositor_glare_linked", bool(report.get("compositor_glare_linked")), 3.0, "CompositorNodeGlare not linked R_LAYERS -> Glare -> COMPOSITE")

    covs = report.get("frame_coverages_pct") or []
    invs = report.get("frame_in_view_ratios") or []
    cov_ok = bool(covs) and all(8.0 <= c <= 92.0 for c in covs)
    inv_ok = bool(invs) and (sum(invs) / len(invs)) >= 0.70
    _chk("framing_coverage", cov_ok and inv_ok, 4.0, f"cov={covs}, in_view={invs}")

    ml = float(px.get("mean_luma", 0.0))
    sl = float(px.get("std_luma", 0.0))
    ee = float(px.get("edge_energy", 0.0))
    cw = float(px.get("pct_clipped_white", 0.0))
    cb = float(px.get("pct_clipped_black", 0.0))
    _chk("exposure_balanced", 0.04 <= ml <= 0.80 and cw <= 15.0 and cb <= 88.0, 3.0, f"luma={ml:.3f}, clip_blk={cb:.1f}%, clip_wht={cw:.1f}%")
    _chk("contrast_and_detail", sl >= 0.04 and ee >= 0.0020, 3.0, f"std_luma={sl:.4f}, edge={ee:.5f}")

    dur = float(report.get("duration_sec", 0.0))
    fps = float(report.get("fps", 0.0))
    _chk("duration_and_fps", 8.0 <= dur <= 14.0 and abs(fps - 24.0) <= 1.0, 2.0, f"dur={dur}s @ {fps}fps (target ~10s @ 24fps)")
    _chk("bezier_keyframes", int(report.get("nonconstant_fcurves", 0)) >= 2 and "BEZIER" in set(report.get("interp_modes") or []), 2.0, f"fcurves={report.get('nonconstant_fcurves',0)}, interp={report.get('interp_modes',[])}")

    tot_motion = float(report.get("cam_motion_m", 0.0)) + float(report.get("hero_motion_m", 0.0))
    fdiff_mean = float(px.get("frame_diff_mean", 0.0))
    fdiff_min = float(px.get("frame_diff_min", 0.0))
    _chk("visible_motion", tot_motion >= 1.0 and fdiff_mean >= 0.015 and fdiff_min >= 0.004, 3.0, f"motion={tot_motion:.2f}m, diff_mean={fdiff_mean:.5f}, diff_min={fdiff_min:.5f}")

    return checks, reasons, round(score, 2)


class CelebrationVerifier:
    """Two-process Blender 4.5.14 LTS verifier + 1080p MP4 renderer."""

    def __init__(
        self,
        verify_blender_bin: str | None = None,
        *,
        render_device: str = "CPU",
        cycles_samples: int = 24,
        res_x: int = 640,
        res_y: int = 360,
        timeout_s: int = 180,
    ) -> None:
        self.verify_blender_bin = resolve_blender_binary(verify_blender_bin)
        self.render_device = render_device.upper()
        self.cycles_samples = cycles_samples
        self.res_x = res_x
        self.res_y = res_y
        self.timeout_s = timeout_s

    def verify_script(
        self,
        script_code: str,
        *,
        save_dir: str,
        prefix: str = "iter_01",
        do_render: bool = True,
    ) -> CelebrationVerification:
        os.makedirs(save_dir, exist_ok=True)
        if not (script_code or "").strip() or "import bpy" not in script_code:
            return CelebrationVerification(
                ok_execution=False,
                host_passed=False,
                host_score=0.0,
                exec_seconds=0.0,
                checks={"zero_traceback": False},
                failed_reasons=["EmptyScriptError: model output contained no executable bpy script (possible max_tokens truncation in reasoning)"],
                traceback_type="EmptyScriptError",
                traceback_summary="EmptyScriptError: empty or missing bpy Python script",
            )
        env = os.environ.copy()
        env["PYTHONDONTWRITEBYTECODE"] = "1"

        with tempfile.TemporaryDirectory(prefix="celeb_bpy_") as tmpdir:
            blend_path = os.path.join(tmpdir, "scene.blend")
            sentinel_path = os.path.join(tmpdir, "exec_sentinel.json")
            render_dir = os.path.join(tmpdir, "renders")
            os.makedirs(render_dir, exist_ok=True)
            report_path = os.path.join(tmpdir, "report.json")
            cfg_path = os.path.join(tmpdir, "cfg.json")

            save_cfg = {"blend_path": blend_path, "sentinel_path": sentinel_path}
            cand_script = (
                script_code.rstrip()
                + "\n\n"
                + f"__HOST_SAVE_CFG__ = {repr(save_cfg)}\n"
                + _SAVE_EPILOGUE_TEMPLATE
            )
            cand_path = os.path.join(tmpdir, "candidate.py")
            with open(cand_path, "w", encoding="utf-8") as fh:
                fh.write(cand_script)

            t0 = time.time()
            try:
                p1 = subprocess.run(
                    [self.verify_blender_bin, "-b", "--factory-startup", "--python-exit-code", "1", "--python", cand_path],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_s,
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return CelebrationVerification(
                    ok_execution=False,
                    host_passed=False,
                    host_score=0.0,
                    exec_seconds=round(time.time() - t0, 2),
                    checks={"zero_traceback": False},
                    failed_reasons=[f"timeout > {self.timeout_s}s"],
                    traceback_type="TimeoutExpired",
                    traceback_summary=f"TimeoutExpired: exceeded {self.timeout_s}s",
                )

            if p1.returncode != 0 or not os.path.isfile(blend_path) or not os.path.isfile(sentinel_path):
                elapsed = round(time.time() - t0, 2)
                exc_type, code_line, summary = _extract_traceback_summary(p1.stdout, p1.stderr)
                if not summary:
                    summary = f"Process 1 exited rc={p1.returncode} without writing scene.blend"
                    exc_type = "RuntimeError"
                return CelebrationVerification(
                    ok_execution=False,
                    host_passed=False,
                    host_score=0.0,
                    exec_seconds=elapsed,
                    checks={"zero_traceback": False},
                    failed_reasons=[summary],
                    traceback_type=exc_type or "RuntimeError",
                    traceback_line=code_line,
                    traceback_summary=summary,
                )

            # Save persistent .blend for the iteration
            saved_blend = os.path.join(save_dir, f"{prefix}.blend")
            shutil.copy2(blend_path, saved_blend)

            vcfg = {
                "blend_path": saved_blend,
                "report_path": report_path,
                "render_dir": render_dir,
                "do_render": bool(do_render),
                "render_device": self.render_device,
                "cycles_samples": self.cycles_samples,
                "use_denoising": True,
                "res_x": self.res_x,
                "res_y": self.res_y,
            }
            with open(cfg_path, "w", encoding="utf-8") as fh:
                json.dump(vcfg, fh)
            vscript = os.path.join(tmpdir, "verify_celeb.py")
            with open(vscript, "w", encoding="utf-8") as fh:
                fh.write(_CELEBRATION_VERIFY_SCRIPT)

            try:
                p2 = subprocess.run(
                    [self.verify_blender_bin, "-b", "--factory-startup", "--python-exit-code", "1", "--python", vscript, "--", cfg_path],
                    capture_output=True,
                    text=True,
                    timeout=max(self.timeout_s, 300),
                    env=env,
                )
            except subprocess.TimeoutExpired:
                return CelebrationVerification(
                    ok_execution=False,
                    host_passed=False,
                    host_score=0.0,
                    exec_seconds=round(time.time() - t0, 2),
                    checks={"zero_traceback": False},
                    failed_reasons=["verifier render timeout"],
                    traceback_type="VerifierTimeout",
                    traceback_summary="Verifier Process 2 timed out",
                    saved_blend=saved_blend,
                )

            elapsed = round(time.time() - t0, 2)
            if p2.returncode != 0 or not os.path.isfile(report_path):
                exc_type, code_line, summary = _extract_traceback_summary(p2.stdout, p2.stderr)
                return CelebrationVerification(
                    ok_execution=False,
                    host_passed=False,
                    host_score=0.0,
                    exec_seconds=elapsed,
                    checks={"zero_traceback": False},
                    failed_reasons=[summary or f"Process 2 failed rc={p2.returncode}"],
                    traceback_type=exc_type or "VerifierError",
                    traceback_line=code_line,
                    traceback_summary=summary or f"Process 2 failed rc={p2.returncode}",
                    saved_blend=saved_blend,
                )

            with open(report_path, "r", encoding="utf-8") as fh:
                scene_report = json.load(fh)
            px = analyze_keyframes(scene_report.get("rendered_files") or [])

            saved_renders: list[str] = []
            for item in scene_report.get("rendered_files") or []:
                src = item["path"]
                if os.path.isfile(src):
                    dst = os.path.join(save_dir, f"{prefix}_f{int(item['frame']):04d}.png")
                    shutil.copy2(src, dst)
                    saved_renders.append(dst)

            checks, reasons, host_score = score_celebration_host(scene_report, px)
            # Core hard gate for host_passed: executable + non-black + camera framing + visible motion + >= 35/50
            hard_ok = all(
                checks.get(k, False)
                for k in ("zero_traceback", "exposure_balanced", "contrast_and_detail", "framing_coverage", "visible_motion")
            ) and host_score >= 35.0

            return CelebrationVerification(
                ok_execution=True,
                host_passed=hard_ok,
                host_score=host_score,
                exec_seconds=elapsed,
                checks=checks,
                failed_reasons=reasons,
                scene_report=scene_report,
                pixel_stats=px,
                saved_renders=saved_renders,
                saved_blend=saved_blend,
            )


def render_final_1080p_mp4(
    *,
    blend_path: str,
    out_mp4: str,
    frames_dir: str,
    blender_bin: str | None = None,
    render_device: str = "CUDA",
    samples: int = 64,
    res_x: int = 1920,
    res_y: int = 1080,
    fps: int = 24,
    crf: int = 17,
) -> dict[str, Any]:
    """Render full 1080p Cycles OIDN animation frames from `blend_path` and encode bt709 H.264 MP4."""
    exe = resolve_blender_binary(blender_bin)
    os.makedirs(frames_dir, exist_ok=True)
    os.makedirs(os.path.dirname(os.path.abspath(out_mp4)), exist_ok=True)

    render_py = f"""
import bpy, os
bpy.ops.wm.open_mainfile(filepath={blend_path!r})
scene = bpy.context.scene
scene.render.engine = "CYCLES"
req_dev = {render_device.upper()!r}
if req_dev == "CUDA":
    try:
        prefs = bpy.context.preferences.addons["cycles"].preferences
        prefs.compute_device_type = "CUDA"
        prefs.get_devices()
        any_gpu = False
        for d in prefs.devices:
            d.use = (d.type == "CUDA")
            any_gpu = any_gpu or (d.type == "CUDA")
        scene.cycles.device = "GPU" if any_gpu else "CPU"
    except Exception:
        scene.cycles.device = "CPU"
else:
    scene.cycles.device = "CPU"
scene.cycles.samples = {int(samples)}
scene.cycles.use_denoising = True
if hasattr(scene.cycles, "denoiser"):
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
scene.render.resolution_x = {int(res_x)}
scene.render.resolution_y = {int(res_y)}
scene.render.resolution_percentage = 100
scene.render.image_settings.file_format = "PNG"
scene.render.filepath = os.path.join({frames_dir!r}, "frame_")
bpy.ops.render.render(animation=True)
print("RENDER_1080P_DONE", scene.frame_start, scene.frame_end, scene.cycles.device)
"""
    t0 = time.time()
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as tf:
        tf.write(render_py)
        script_path = tf.name
    try:
        subprocess.run(
            [exe, "-b", "--factory-startup", "--python-exit-code", "1", "--python", script_path],
            check=True,
        )
    finally:
        try:
            os.unlink(script_path)
        except OSError:
            pass

    pngs = sorted(f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png"))
    if not pngs:
        raise RuntimeError(f"No PNG frames rendered into {frames_dir}")

    # Determine start number from first rendered frame
    first_num = int(pngs[0].replace("frame_", "").replace(".png", ""))
    titled_dir = frames_dir.rstrip("/") + "_titled"
    n_final_frames = apply_editorial_titles_and_credits(
        src_frames_dir=frames_dir,
        dst_frames_dir=titled_dir,
        fps=fps,
    )
    ffmpeg_cmd = [
        shutil.which("ffmpeg") or "/home/orwel/.local/bin/ffmpeg",
        "-y", "-v", "error",
        "-framerate", str(int(fps)),
        "-start_number", "1",
        "-i", os.path.join(titled_dir, "frame_%04d.png"),
        "-c:v", "libx264", "-preset", "slow", "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p",
        "-colorspace", "bt709", "-color_primaries", "bt709", "-color_trc", "bt709",
        "-movflags", "+faststart",
        out_mp4,
    ]
    subprocess.run(ffmpeg_cmd, check=True)
    return {
        "out_mp4": out_mp4,
        "n_scene_frames": len(pngs),
        "n_final_frames": n_final_frames,
        "first_scene_frame": first_num,
        "bytes": os.path.getsize(out_mp4),
        "render_and_encode_seconds": round(time.time() - t0, 2),
    }


def _load_font(bold: bool, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    cand = (
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
        if bold
        else "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"
    )
    if os.path.isfile(cand):
        return ImageFont.truetype(cand, size)
    return ImageFont.load_default()


def apply_editorial_titles_and_credits(
    *,
    src_frames_dir: str,
    dst_frames_dir: str,
    fps: int = 24,
    end_card_seconds: float = 4.5,
) -> int:
    """Composite Atlas Brand Guide p.19 editorial chapter titles + open-source tribute end card.

    Palette: #101518 dark ground, #82C232 green, #556D79 blue steel, #686868 dark grey,
    #E0E0E0 light grey, #FFFFFF white. Bold ALL-CAPS headings, thick green accent bars,
    '//' technical tags, hairline dividers, Liberation Sans, quiet small footer credit only.
    """
    os.makedirs(dst_frames_dir, exist_ok=True)
    pngs = sorted(f for f in os.listdir(src_frames_dir) if f.startswith("frame_") and f.endswith(".png"))
    if not pngs:
        raise RuntimeError(f"No frames in {src_frames_dir}")

    sample = Image.open(os.path.join(src_frames_dir, pngs[0])).convert("RGBA")
    W, H = sample.size
    scale = H / 1080.0
    f_kicker = _load_font(True, max(11, int(round(19 * scale))))
    f_title = _load_font(True, max(16, int(round(34 * scale))))
    f_end_h = _load_font(True, max(18, int(round(40 * scale))))
    f_end_sub = _load_font(True, max(13, int(round(23 * scale))))
    f_end_cred = _load_font(False, max(12, int(round(21 * scale))))
    f_end_foot = _load_font(False, max(11, int(round(17 * scale))))

    n_scene = len(pngs)
    fade_out_frames = max(1, int(round(0.8 * fps)))
    out_idx = 0

    def _alpha_envelope(idx: int, start_f: int, end_f: int, ramp: int = 12) -> float:
        if idx < start_f or idx > end_f:
            return 0.0
        a_in = min(1.0, (idx - start_f) / max(1, ramp))
        a_out = min(1.0, (end_f - idx) / max(1, ramp))
        return max(0.0, min(a_in, a_out))

    ch1_end = int(round(n_scene * 0.42))
    ch2_start = int(round(n_scene * 0.48))
    ch2_end = int(round(n_scene * 0.90))

    # Brand Guide p.19 exact RGBs
    COL_BG = (16, 21, 24)       # #101518
    COL_GREEN = (130, 194, 50)  # #82C232
    COL_STEEL = (85, 109, 121)  # #556D79
    COL_DGREY = (104, 104, 104) # #686868
    COL_LGREY = (224, 224, 224) # #E0E0E0
    COL_WHITE = (255, 255, 255) # #FFFFFF

    for i, fn in enumerate(pngs):
        im = Image.open(os.path.join(src_frames_dir, fn)).convert("RGBA")
        overlay = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(overlay)

        # Chapter 1: 10 Years of NVIDIA AI Architecture (with subtle #101518 legibility scrim)
        a1 = _alpha_envelope(i, int(0.4 * fps), ch1_end, ramp=int(0.5 * fps))
        if a1 > 0.01:
            pad_x, pad_y = int(56 * scale), int(48 * scale)
            bar_w = max(3, int(7 * scale))
            d.rectangle(
                [pad_x - int(14 * scale), H - pad_y - int(96 * scale), pad_x + int(760 * scale), H - pad_y + int(12 * scale)],
                fill=(*COL_BG, int(175 * a1)),
            )
            d.rectangle([pad_x, H - pad_y - int(84 * scale), pad_x + bar_w, H - pad_y], fill=(*COL_GREEN, int(245 * a1)))
            d.text((pad_x + int(22 * scale), H - pad_y - int(78 * scale)), "// 2016 PASCAL -> 2026 BLACKWELL", font=f_kicker, fill=(*COL_GREEN, int(245 * a1)))
            d.text((pad_x + int(22 * scale), H - pad_y - int(44 * scale)), "10 YEARS OF NVIDIA AI ARCHITECTURE", font=f_title, fill=(*COL_WHITE, int(245 * a1)))

        # Chapter 2: Thinking Machines Lab · Inkling-2-Mini & Open-Source Centaurs
        a2 = _alpha_envelope(i, ch2_start, ch2_end, ramp=int(0.5 * fps))
        if a2 > 0.01:
            pad_x, pad_y = int(56 * scale), int(48 * scale)
            bar_w = max(3, int(7 * scale))
            d.rectangle(
                [pad_x - int(14 * scale), H - pad_y - int(96 * scale), pad_x + int(980 * scale), H - pad_y + int(12 * scale)],
                fill=(*COL_BG, int(175 * a2)),
            )
            d.rectangle([pad_x, H - pad_y - int(84 * scale), pad_x + bar_w, H - pad_y], fill=(*COL_GREEN, int(245 * a2)))
            d.text((pad_x + int(22 * scale), H - pad_y - int(78 * scale)), "// THINKING MACHINES LAB  ·  OPEN SOURCE + HUMAN-AI CENTAURS", font=f_kicker, fill=(*COL_GREEN, int(245 * a2)))
            d.text((pad_x + int(22 * scale), H - pad_y - int(44 * scale)), "INKLING-2-MINI  ·  VERIFIED LEDGER + NATIVE GPU VISION", font=f_title, fill=(*COL_WHITE, int(245 * a2)))

        comp = Image.alpha_composite(im, overlay).convert("RGB")
        if i >= n_scene - fade_out_frames:
            t = (i - (n_scene - fade_out_frames) + 1) / float(fade_out_frames)
            bg = Image.new("RGB", (W, H), COL_BG)
            comp = Image.blend(comp, bg, min(1.0, max(0.0, t)))
        out_idx += 1
        comp.save(os.path.join(dst_frames_dir, f"frame_{out_idx:04d}.png"))

    # Closing Editorial Tribute Card (#101518 ground, thick #82C232 accent rule, hairline #556D79 dividers)
    n_end = max(1, int(round(end_card_seconds * fps)))
    card_bg = Image.new("RGB", (W, H), COL_BG)
    cx = W // 2
    credit_lines = [
        ("NVIDIA", "10 Years of AI Architecture (2016 Pascal -> 2026 Blackwell)", COL_GREEN),
        ("THINKING MACHINES LAB", "Inkling-2-Mini", COL_WHITE),
        ("SATELLITEDOWN/INGINE", "with fngine · cinference · NInfer", COL_LGREY),
        ("GGML-ORG/LLAMA.CPP  ·  REALUGBUN  ·  MW00/PROJECT-MAYA", "open inference foundations across NVIDIA & AMD", COL_LGREY),
        ("TURBODERP/EXLLAMAV3  ·  PROFESSORPALMER/KINTSUGI", "quantization & precision craftsmanship", COL_LGREY),
    ]
    for j in range(n_end):
        a = _alpha_envelope(j, 0, n_end - 1, ramp=int(0.55 * fps))
        frame = card_bg.copy().convert("RGBA")
        ov = Image.new("RGBA", (W, H), (0, 0, 0, 0))
        d = ImageDraw.Draw(ov)

        y = int(190 * scale)
        d.text((cx, y), "// OPEN SOURCE + HUMAN-AI COLLABORATION", font=f_kicker, fill=(*COL_GREEN, int(240 * a)), anchor="mm")
        y += int(52 * scale)
        d.text((cx, y), "CELEBRATING A DECADE OF SILICON & OPEN CRAFT", font=f_end_h, fill=(*COL_WHITE, int(250 * a)), anchor="mm")
        y += int(38 * scale)
        # Thick green accent bar centered over subtle blue-steel hairline
        d.rectangle([cx - int(360 * scale), y + max(1, int(2 * scale)), cx + int(360 * scale), y + max(2, int(3 * scale))], fill=(*COL_STEEL, int(180 * a)))
        d.rectangle([cx - int(110 * scale), y, cx + int(110 * scale), y + max(3, int(6 * scale))], fill=(*COL_GREEN, int(245 * a)))

        y += int(64 * scale)
        for name, desc, col in credit_lines:
            d.text((cx, y), name, font=f_end_sub, fill=(*col, int(245 * a)), anchor="mm")
            y += int(31 * scale)
            d.text((cx, y), desc, font=f_end_cred, fill=(*COL_LGREY, int(205 * a)), anchor="mm")
            y += int(52 * scale)

        # Hairline divider above quiet footer
        foot_y = H - int(124 * scale)
        d.rectangle([cx - int(420 * scale), foot_y, cx + int(420 * scale), foot_y + max(1, int(1 * scale))], fill=(*COL_STEEL, int(140 * a)))
        d.text(
            (cx, H - int(90 * scale)),
            "What open source and human-AI ('monkey centaur') teams can build together — zero monetization, pure craft.",
            font=f_end_foot,
            fill=(*COL_LGREY, int(210 * a)),
            anchor="mm",
        )
        d.text(
            (cx, H - int(58 * scale)),
            "ATLAS 3DSS  //  3D scene hillclimbed in Blender 4.5 LTS via Verified Ledger + Native GPU Vision (Inkling-2-Mini on atlas-ingine)",
            font=f_end_foot,
            fill=(*COL_DGREY, int(200 * a)),
            anchor="mm",
        )
        out_idx += 1
        Image.alpha_composite(frame, ov).convert("RGB").save(os.path.join(dst_frames_dir, f"frame_{out_idx:04d}.png"))
    return out_idx


def build_progression_sheet(
    rows: list[tuple[str, list[str]]],
    out_png: str,
    *,
    thumb_w: int = 384,
    thumb_h: int = 216,
) -> str:
    """Stitch labeled multi-timestamp keyframe rows (e.g. Iter 1 vs Mid vs Champion) into one PNG."""
    valid_rows = [(lbl, [p for p in paths if os.path.isfile(p)][:5]) for lbl, paths in rows if paths]
    if not valid_rows:
        return ""
    ncols = max(len(ps) for _, ps in valid_rows)
    header_h = 34
    pad = 10
    w = ncols * (thumb_w + pad) + pad
    h = len(valid_rows) * (thumb_h + header_h + pad) + pad
    canvas = Image.new("RGB", (w, h), (16, 21, 24))
    draw = ImageDraw.Draw(canvas)
    hdr_font = _load_font(True, 16)

    for r_idx, (lbl, paths) in enumerate(valid_rows):
        y0 = pad + r_idx * (thumb_h + header_h + pad)
        draw.text((pad + 4, y0 + 8), lbl.replace("—", "-"), font=hdr_font, fill=(224, 224, 224))
        for c_idx, p in enumerate(paths):
            x0 = pad + c_idx * (thumb_w + pad)
            im = Image.open(p).convert("RGB").resize((thumb_w, thumb_h), Image.Resampling.LANCZOS)
            canvas.paste(im, (x0, y0 + header_h))
    os.makedirs(os.path.dirname(os.path.abspath(out_png)), exist_ok=True)
    canvas.save(out_png)
    return out_png

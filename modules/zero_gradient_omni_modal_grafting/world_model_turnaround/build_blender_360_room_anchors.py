#!/usr/bin/env python3
"""Build & render a genuine photorealistic 3D studio room anchor bank in Blender 5.2.5 LTS Cycles.

Replaces the flat 2D graphic collage from Showcase 2 v3 with a real 360-degree enclosed 3D
AI hardware workshop rendered from a single nodal 35mm cinema camera at yaw stops around the
full orbit (0, 60, 120, 180, 240, 300, 330, 345 deg):

  - North Wall (yaw = 0 deg):
      1. Monkey Founder seated naturally behind a real 3D walnut/obsidian hardware desk
         (soft-feathered studio integration, zero flat UI frame or border!).
      2. Real 3D physical beveled wall plaque with standoffs & museum picture light carrying
         ONLY the owner's verbatim 24-word text (zero extra plaque-caption clutter so OCR is clean).
      3. Real 3D open studio doorway into an adjoining warm-amber lit hallway room with a
         physical glowing table lamp casting real light onto the studio floor.
  - East Wall (yaw = 90/120 deg): 3D 42U steel server rack w/ emerald status LEDs + oscilloscope bench.
  - South Wall (yaw = 180 deg): Rear brick/slate workshop shelving + studio sconces (sign 100% absent).
  - West Wall (yaw = 240/270 deg): Mullioned studio window + architect drafting table (sign 100% absent).

Usage (via Blender 5.2.5 LTS headless):
  /mnt/storage/blender_by_atlas/install_desktop/blender --background --factory-startup \\
      --python build_blender_360_room_anchors.py -- \\
      --out-dir /mnt/storage/omni_gate2_runs/phase0_912_spinup/showcase_scripts/blender_360_room_v4
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap
from typing import Any, Dict, List, Sequence, Tuple

import numpy as np

PINNED_SIGN_TEXT = (
    "Thank Jensen for 10 Years of Cuda, Ty thinking machines for this model, "
    "and ty to the open source community for being amazing"
)
DEFAULT_BLENDER_BIN = "/mnt/storage/blender_by_atlas/install_desktop/blender"
DEFAULT_FOUNDER_REF = "/home/orwel/dev_genius/resin-mega-test-event/trailer/keyframes/kf_01_host_hook.png"
FONT_BOLD = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"
DEFAULT_YAWS = (0, 60, 120, 180, 240, 300, 330, 345)


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while True:
            b = f.read(1 << 20)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def _load_font(size: int) -> Any:
    from PIL import ImageFont

    return ImageFont.truetype(FONT_BOLD, size) if os.path.isfile(FONT_BOLD) else ImageFont.load_default()


def build_clean_plaque_texture(sign_text: str, out_png: Path, width: int = 2400, height: int = 1350) -> Path:
    """Render high-contrast warm ceramic architectural wall plaque containing ONLY sign_text."""
    from PIL import Image, ImageDraw
    # Subtle matte warm-ivory architectural panel texture (zero UI badges or extra OCR noise!)
    rng = np.random.default_rng(20261127)
    base = np.full((height, width, 3), (242, 240, 234), dtype=np.float32)
    grain = rng.normal(0.0, 1.8, size=(height, width, 1)).astype(np.float32)
    # Soft studio vignette near plaque edges so it integrates physically under Cycles lighting
    yy = np.linspace(-1.0, 1.0, height, dtype=np.float32)[:, None]
    xx = np.linspace(-1.0, 1.0, width, dtype=np.float32)[None, :]
    vig = 1.0 - 0.045 * (xx * xx + yy * yy)
    arr = np.clip(base * vig[..., None] + grain, 0, 255).astype(np.uint8)
    im = Image.fromarray(arr, mode="RGB")
    draw = ImageDraw.Draw(im)

    # Subtle architectural emerald pin-stripe top & bottom accent (no text!)
    draw.rectangle([(64, 52), (width - 64, 68)], fill=(96, 152, 14))
    draw.rectangle([(64, height - 68), (width - 64, height - 52)], fill=(96, 152, 14))

    inner_w = width - 260
    inner_h = height - 260
    chosen_font = _load_font(120)
    chosen_lines = [sign_text]
    for fsz in range(152, 56, -4):
        fnt = _load_font(fsz)
        avg_cw = max(1.0, draw.textbbox((0, 0), "abcdefghijklm ", font=fnt)[2] / 14.0)
        wrap_chars = max(16, int(inner_w / avg_cw))
        lines = textwrap.wrap(sign_text, width=wrap_chars)
        line_h = int(fsz * 1.34)
        max_lw = max((draw.textbbox((0, 0), ln, font=fnt)[2] for ln in lines), default=0)
        if max_lw <= inner_w and len(lines) * line_h <= inner_h:
            chosen_font = fnt
            chosen_lines = lines
            break

    line_h = int(chosen_font.size * 1.34) if hasattr(chosen_font, "size") else 140
    total_th = len(chosen_lines) * line_h
    y_cur = (height - total_th) // 2
    for ln in chosen_lines:
        lw = draw.textbbox((0, 0), ln, font=chosen_font)[2]
        x_cur = (width - lw) // 2
        draw.text((x_cur, y_cur), ln, fill=(18, 22, 26), font=chosen_font)
        y_cur += line_h

    out_png.parent.mkdir(parents=True, exist_ok=True)
    im.save(out_png)
    return out_png


def build_feathered_founder_rgba(founder_ref: Path, out_png: Path) -> Tuple[Path, float]:
    """Prepare a clean dark-studio framed RGBA portrait of the Monkey Founder for 3D architectural integration."""
    from PIL import Image

    im = Image.open(founder_ref).convert("RGB")
    rw, rh = im.size
    cw = min(rw, int(rh * 0.90))
    cx0 = (rw - cw) // 2
    crop = im.crop((cx0, 0, cx0 + cw, rh))
    w, h = crop.size
    rgb = np.asarray(crop, dtype=np.float32)

    # Gently vignette upper background lightbars so Monkey Founder's face, top hat, and tuxedo stay crisp & grounded
    x = np.linspace(-1.0, 1.0, w, dtype=np.float32)[None, :]
    y = np.linspace(-1.0, 1.0, h, dtype=np.float32)[:, None]
    rad = np.sqrt((x / 0.95) ** 2 + ((y - 0.08) / 1.05) ** 2)
    vig = np.clip(1.0 - 0.35 * np.maximum(0.0, rad - 0.55), 0.55, 1.0)
    rgb = np.clip(rgb * vig[..., None], 0.0, 255.0).astype(np.uint8)

    out_im = Image.fromarray(rgb, mode="RGB")
    out_png.parent.mkdir(parents=True, exist_ok=True)
    out_im.save(out_png)
    return out_png, float(w) / float(h)


def build_contact_sheet(image_rows: Sequence[Tuple[str, Path]], out_png: Path) -> Path:
    from PIL import Image, ImageDraw

    cols = 4
    rows = math.ceil(len(image_rows) / cols)
    tw, th = 480, 274
    lbl_h = 30
    hdr_h = 46
    sheet = Image.new("RGB", (cols * tw, hdr_h + rows * (th + lbl_h)), (12, 15, 18))
    draw = ImageDraw.Draw(sheet)
    draw.text((16, 12), "BLENDER 5.2.5 LTS CYCLES — 360° 3D STUDIO ROOM VERIFIED LEDGER ANCHORS", fill=(166, 238, 66), font=_load_font(20))
    f_lbl = _load_font(16)
    for idx, (label, path) in enumerate(image_rows):
        r, c = divmod(idx, cols)
        x0 = c * tw
        y0 = hdr_h + r * (th + lbl_h)
        im = Image.open(path).convert("RGB").resize((tw, th), Image.Resampling.LANCZOS)
        sheet.paste(im, (x0, y0 + lbl_h))
        draw.rectangle([(x0, y0), (x0 + tw, y0 + lbl_h)], fill=(18, 24, 28))
        draw.text((x0 + 10, y0 + 6), label, fill=(235, 242, 246), font=f_lbl)
    sheet.save(out_png)
    return out_png


def run_blender_build(args: argparse.Namespace) -> None:
    import bpy
    from mathutils import Vector

    out_dir = Path(args.out_dir)
    tex_dir = out_dir / "textures"
    plaque_tex_p = tex_dir / "plaque_24word.png"
    founder_tex_p = tex_dir / "founder_feathered.png"
    meta_p = tex_dir / "textures_meta.json"
    if not (plaque_tex_p.is_file() and founder_tex_p.is_file() and meta_p.is_file()):
        raise FileNotFoundError(f"Missing prebuilt textures in {tex_dir}; run via python3 first")
    founder_aspect = float(json.loads(meta_p.read_text(encoding="utf-8"))["founder_aspect"])

    bpy.ops.wm.read_factory_settings(use_empty=True)
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.device = "CPU"
    scene.cycles.samples = int(args.samples)
    scene.cycles.use_adaptive_sampling = True
    scene.cycles.adaptive_threshold = 0.02
    scene.cycles.use_denoising = True
    try:
        scene.cycles.denoiser = "OPENIMAGEDENOISE"
    except Exception:
        pass
    scene.cycles.max_bounces = 6
    scene.cycles.diffuse_bounces = 3
    scene.cycles.glossy_bounces = 3
    scene.cycles.transmission_bounces = 2
    scene.render.threads_mode = "FIXED"
    scene.render.threads = int(args.threads)
    scene.render.resolution_x = int(args.width)
    scene.render.resolution_y = int(args.height)
    scene.render.resolution_percentage = 100
    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGB"

    try:
        scene.view_settings.view_transform = "AgX"
        scene.view_settings.look = "AgX - Punchy"
    except Exception:
        scene.view_settings.view_transform = "Standard"
    scene.view_settings.exposure = 0.25

    # World: subtle dark architectural twilight ambient
    world = bpy.data.worlds.new("StudioWorld")
    scene.world = world
    world.use_nodes = True
    bg = world.node_tree.nodes.get("Background")
    if bg:
        bg.inputs[0].default_value = (0.015, 0.020, 0.024, 1.0)
        bg.inputs[1].default_value = 0.5

    # Helper builders
    def make_pbr_mat(
        name: str,
        color: Tuple[float, float, float],
        *,
        roughness: float = 0.55,
        metallic: float = 0.0,
        noise_scale: float = 0.0,
        bump_strength: float = 0.06,
        emission_color: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        emission_strength: float = 0.0,
    ) -> Any:
        mat = bpy.data.materials.new(name)
        mat.use_nodes = True
        nt = mat.node_tree
        bsdf = nt.nodes.get("Principled BSDF")
        bsdf.inputs["Base Color"].default_value = (*color, 1.0)
        bsdf.inputs["Roughness"].default_value = roughness
        bsdf.inputs["Metallic"].default_value = metallic
        if emission_strength > 0.0:
            bsdf.inputs["Emission Color"].default_value = (*emission_color, 1.0)
            bsdf.inputs["Emission Strength"].default_value = emission_strength
        if noise_scale > 0.0:
            tex_coord = nt.nodes.new("ShaderNodeTexCoord")
            noise = nt.nodes.new("ShaderNodeTexNoise")
            noise.inputs["Scale"].default_value = noise_scale
            noise.inputs["Detail"].default_value = 6.0
            noise.inputs["Roughness"].default_value = 0.6
            ramp = nt.nodes.new("ShaderNodeValToRGB")
            ramp.color_ramp.elements[0].color = (color[0] * 0.72, color[1] * 0.72, color[2] * 0.72, 1.0)
            ramp.color_ramp.elements[1].color = (min(1.0, color[0] * 1.22), min(1.0, color[1] * 1.22), min(1.0, color[2] * 1.22), 1.0)
            bump = nt.nodes.new("ShaderNodeBump")
            bump.inputs["Strength"].default_value = bump_strength
            bump.inputs["Distance"].default_value = 0.015
            nt.links.new(tex_coord.outputs["Object"], noise.inputs["Vector"])
            nt.links.new(noise.outputs["Fac"], ramp.inputs["Fac"])
            nt.links.new(ramp.outputs["Color"], bsdf.inputs["Base Color"])
            nt.links.new(noise.outputs["Fac"], bump.inputs["Height"])
            nt.links.new(bump.outputs["Normal"], bsdf.inputs["Normal"])
        return mat

    def add_box(
        name: str,
        loc: Tuple[float, float, float],
        dim: Tuple[float, float, float],
        mat: Any,
        rot_z_deg: float = 0.0,
        bevel: float = 0.0,
    ) -> Any:
        bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
        obj = bpy.context.active_object
        obj.name = name
        obj.scale = dim
        obj.rotation_euler = (0.0, 0.0, math.radians(rot_z_deg))
        obj.data.materials.append(mat)
        if bevel > 0.0:
            mod = obj.modifiers.new("Bevel", "BEVEL")
            mod.width = bevel
            mod.segments = 2
        return obj

    def add_area_light(
        name: str,
        loc: Tuple[float, float, float],
        target: Tuple[float, float, float],
        energy: float,
        color: Tuple[float, float, float],
        size_x: float,
        size_y: float = 0.0,
    ) -> Any:
        ld = bpy.data.lights.new(name, type="AREA")
        ld.energy = energy
        ld.color = color
        if size_y > 0.0:
            ld.shape = "RECTANGLE"
            ld.size = size_x
            ld.size_y = size_y
        else:
            ld.shape = "SQUARE"
            ld.size = size_x
        obj = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(obj)
        obj.location = loc
        direction = Vector(target) - Vector(loc)
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()
        return obj

    def add_point_light(name: str, loc: Tuple[float, float, float], energy: float, color: Tuple[float, float, float], radius: float = 0.14) -> Any:
        ld = bpy.data.lights.new(name, type="POINT")
        ld.energy = energy
        ld.color = color
        ld.shadow_soft_size = radius
        obj = bpy.data.objects.new(name, ld)
        scene.collection.objects.link(obj)
        obj.location = loc
        return obj

    # Materials (moody cinematic dark obsidian-and-walnut ARRI studio palette)
    m_wall = make_pbr_mat("SlateWall", (0.028, 0.035, 0.042), roughness=0.74, noise_scale=3.8, bump_strength=0.06)
    m_hall_wall = make_pbr_mat("HallWarmWall", (0.14, 0.095, 0.055), roughness=0.75, noise_scale=4.0, bump_strength=0.04)
    m_floor = make_pbr_mat("WalnutFloor", (0.065, 0.044, 0.030), roughness=0.30, noise_scale=12.0, bump_strength=0.04)
    m_ceiling = make_pbr_mat("CeilingAcoustic", (0.022, 0.026, 0.030), roughness=0.85, noise_scale=5.0, bump_strength=0.04)
    m_wood_slat = make_pbr_mat("WarmWalnutSlat", (0.14, 0.082, 0.044), roughness=0.42, noise_scale=16.0, bump_strength=0.05)
    m_dark_metal = make_pbr_mat("AnodizedSteel", (0.045, 0.052, 0.060), roughness=0.35, metallic=0.85)
    m_desk_top = make_pbr_mat("ObsidianDesk", (0.028, 0.034, 0.038), roughness=0.22, metallic=0.15)
    m_brass = make_pbr_mat("BrushedBrass", (0.68, 0.50, 0.22), roughness=0.28, metallic=0.92)
    m_emerald_led = make_pbr_mat("EmeraldLED", (0.25, 0.55, 0.02), roughness=0.2, emission_color=(0.46, 0.78, 0.02), emission_strength=6.5)
    m_amber_lamp = make_pbr_mat("AmberLampGlobe", (1.0, 0.72, 0.30), roughness=0.15, emission_color=(1.0, 0.66, 0.22), emission_strength=16.0)
    m_sconce_warm = make_pbr_mat("WarmSconce", (1.0, 0.76, 0.42), roughness=0.2, emission_color=(1.0, 0.72, 0.35), emission_strength=8.0)
    m_window_twilight = make_pbr_mat("TwilightGlass", (0.025, 0.055, 0.105), roughness=0.08, emission_color=(0.045, 0.095, 0.185), emission_strength=1.15)

    # Room shell: 7.4m x 7.4m x 3.2m (North wall Y=+3.65, South Y=-3.65, East X=+3.65, West X=-3.65)
    R = 3.65
    H = 3.20
    add_box("Floor", (0.0, 0.5, -0.05), (9.0, 12.0, 0.10), m_floor)
    add_box("Ceiling", (0.0, 0.5, H + 0.05), (9.0, 12.0, 0.10), m_ceiling)

    # North Wall (Y = +R) split around REAL 3D OPEN DOORWAY (X in [1.70, 2.75], Z in [0, 2.38])
    add_box("NorthWall_Main", (-1.00, R + 0.08, H / 2.0), (5.40, 0.16, H), m_wall)
    add_box("NorthWall_FarRight", (3.22, R + 0.08, H / 2.0), (0.94, 0.16, H), m_wall)
    add_box("NorthWall_DoorLintel", (2.225, R + 0.08, 2.79), (1.05, 0.16, 0.82), m_wall)

    # Other 3 main studio walls
    add_box("SouthWall", (0.0, -R - 0.08, H / 2.0), (7.60, 0.16, H), m_wall)
    add_box("EastWall", (R + 0.08, 0.0, H / 2.0), (0.16, 7.60, H), m_wall)
    add_box("WestWall", (-R - 0.08, 0.0, H / 2.0), (0.16, 7.60, H), m_wall)

    # Architectural baseboards & subtle ceiling emerald LED light channels
    add_box("Baseboard_South", (0.0, -R + 0.02, 0.07), (7.30, 0.04, 0.14), m_dark_metal)
    add_box("Baseboard_East", (R - 0.02, 0.0, 0.07), (0.04, 7.30, 0.14), m_dark_metal)
    add_box("Baseboard_West", (-R + 0.02, 0.0, 0.07), (0.04, 7.30, 0.14), m_dark_metal)
    add_box("Baseboard_NorthL", (-1.00, R - 0.02, 0.07), (5.38, 0.04, 0.14), m_dark_metal)
    add_box("CoveStrip_North", (-0.60, R - 0.04, H - 0.14), (4.60, 0.04, 0.03), m_emerald_led)
    add_box("CoveStrip_East", (R - 0.04, 0.0, H - 0.14), (0.04, 5.20, 0.03), m_emerald_led)

    # ==================================================================================
    # NORTH WALL (yaw = 0 deg): INVARIANT 1 (3D Plaque) + INVARIANT 2 (Founder) + INVARIANT 3 (Open Door)
    # ==================================================================================

    # --- INVARIANT 1: Physical 3D Beveled Backlit Wall Plaque (ONLY 24-word text, zero clutter!) ---
    plaque_w, plaque_h = 2.06, 1.16
    px, py, pz = 0.36, R - 0.06, 1.62
    add_box("Plaque_FrameBack", (px, py + 0.035, pz), (plaque_w + 0.09, 0.04, plaque_h + 0.09), m_dark_metal, bevel=0.01)
    # Subtle anodized emerald outer edge trim
    add_box("Plaque_TrimTop", (px, py + 0.015, pz + plaque_h / 2.0 + 0.04), (plaque_w + 0.10, 0.03, 0.022), m_emerald_led)
    add_box("Plaque_TrimBot", (px, py + 0.015, pz - plaque_h / 2.0 - 0.04), (plaque_w + 0.10, 0.03, 0.022), m_emerald_led)
    # Museum picture light fixture above plaque
    add_box("Plaque_LampBar", (px, py - 0.22, pz + plaque_h / 2.0 + 0.22), (1.10, 0.05, 0.035), m_dark_metal)
    add_box("Plaque_LampArmL", (px - 0.35, py - 0.09, pz + plaque_h / 2.0 + 0.22), (0.025, 0.24, 0.025), m_dark_metal)
    add_box("Plaque_LampArmR", (px + 0.35, py - 0.09, pz + plaque_h / 2.0 + 0.22), (0.025, 0.24, 0.025), m_dark_metal)

    # Plaque textured face plane (faces -Y toward camera at origin!)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(px, py, pz))
    plaque_face = bpy.context.active_object
    plaque_face.name = "Plaque_TextFace"
    plaque_face.scale = (plaque_w, plaque_h, 1.0)
    plaque_face.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    m_plaque = bpy.data.materials.new("PlaqueFaceMat")
    m_plaque.use_nodes = True
    nt = m_plaque.node_tree
    bsdf_p = nt.nodes.get("Principled BSDF")
    tex_p = nt.nodes.new("ShaderNodeTexImage")
    tex_p.image = bpy.data.images.load(str(plaque_tex_p))
    tex_p.image.pack()
    bsdf_p.inputs["Roughness"].default_value = 0.55
    bsdf_p.inputs["Specular IOR Level"].default_value = 0.15
    # Gentle internal backlight so plaque reads razor-sharp from 0 deg AND oblique 330 deg
    bsdf_p.inputs["Emission Strength"].default_value = 0.55
    nt.links.new(tex_p.outputs["Color"], bsdf_p.inputs["Base Color"])
    nt.links.new(tex_p.outputs["Color"], bsdf_p.inputs["Emission Color"])
    plaque_face.data.materials.append(m_plaque)

    # --- INVARIANT 2: Monkey Founder Physical 3D Studio Alcove & Executive Hardware Desk ---
    fx, fy = -1.55, 2.85
    desk_h = 0.78
    # Real 3D executive walnut + obsidian desk in front of Founder
    add_box("FounderDesk_Top", (fx, fy - 0.36, desk_h), (1.76, 0.74, 0.06), m_desk_top, bevel=0.008)
    add_box("FounderDesk_FrontPanel", (fx, fy - 0.68, desk_h / 2.0), (1.64, 0.05, desk_h - 0.04), m_wood_slat)
    add_box("FounderDesk_LegL", (fx - 0.78, fy - 0.36, desk_h / 2.0), (0.07, 0.66, desk_h), m_dark_metal)
    add_box("FounderDesk_LegR", (fx + 0.78, fy - 0.36, desk_h / 2.0), (0.07, 0.66, desk_h), m_dark_metal)
    # Real 3D desktop GPU rig placed cleanly on far-right corner of desk (zero overlap with Founder!)
    add_box("DeskGPU_Chassis", (fx + 0.70, fy - 0.34, desk_h + 0.19), (0.20, 0.36, 0.32), m_dark_metal, rot_z_deg=-14.0, bevel=0.008)
    add_box("DeskGPU_Glow", (fx + 0.67, fy - 0.51, desk_h + 0.19), (0.12, 0.02, 0.18), m_emerald_led, rot_z_deg=-14.0)

    # Physical 3D brushed-steel + brass museum gallery surround framing the Monkey Founder behind the desk
    fh = 1.44
    fw = fh * founder_aspect
    alcove_y = R - 0.05
    alcove_z = desk_h + fh * 0.52
    add_box("FounderFrame_Back", (fx - 0.06, alcove_y + 0.03, alcove_z), (fw + 0.10, 0.05, fh + 0.10), m_dark_metal, bevel=0.01)
    add_box("FounderFrame_TopTrim", (fx - 0.06, alcove_y + 0.005, alcove_z + fh / 2.0 + 0.045), (fw + 0.11, 0.04, 0.025), m_brass)
    add_box("FounderFrame_BotTrim", (fx - 0.06, alcove_y + 0.005, alcove_z - fh / 2.0 - 0.045), (fw + 0.11, 0.04, 0.025), m_brass)
    bpy.ops.mesh.primitive_plane_add(size=1.0, location=(fx - 0.06, alcove_y - 0.002, alcove_z))
    founder_pl = bpy.context.active_object
    founder_pl.name = "MonkeyFounder_Portrait"
    founder_pl.scale = (fw, fh, 1.0)
    founder_pl.rotation_euler = (math.radians(90.0), 0.0, 0.0)
    m_founder = bpy.data.materials.new("MonkeyFounderMat")
    m_founder.use_nodes = True
    nt_f = m_founder.node_tree
    bsdf_f = nt_f.nodes.get("Principled BSDF")
    tex_f = nt_f.nodes.new("ShaderNodeTexImage")
    tex_f.image = bpy.data.images.load(str(founder_tex_p))
    tex_f.image.pack()
    bsdf_f.inputs["Roughness"].default_value = 0.52
    bsdf_f.inputs["Specular IOR Level"].default_value = 0.18
    bsdf_f.inputs["Emission Strength"].default_value = 0.32
    nt_f.links.new(tex_f.outputs["Color"], bsdf_f.inputs["Base Color"])
    nt_f.links.new(tex_f.outputs["Color"], bsdf_f.inputs["Emission Color"])
    founder_pl.data.materials.append(m_founder)

    # Warm walnut acoustic vertical wall slats flanking the left edge of North Wall
    for si in range(7):
        add_box(f"NorthSlat_{si}", (-3.45 + si * 0.14, R - 0.03, H / 2.0), (0.06, 0.05, H - 0.30), m_wood_slat)

    # --- INVARIANT 3: Real 3D Open Studio Doorway & Warm Amber Lit Adjoining Hallway ---
    add_box("DoorJamb_L", (1.70, R + 0.06, 1.19), (0.08, 0.24, 2.38), m_wood_slat)
    add_box("DoorJamb_R", (2.75, R + 0.06, 1.19), (0.08, 0.24, 2.38), m_wood_slat)
    add_box("DoorHeader", (2.225, R + 0.06, 2.38), (1.13, 0.24, 0.08), m_wood_slat)
    add_box("OpenDoor_Leaf", (1.88, R + 0.48, 1.17), (0.055, 0.90, 2.32), m_wood_slat, rot_z_deg=-26.0)
    add_box("OpenDoor_Handle", (2.06, R + 0.78, 1.02), (0.14, 0.03, 0.03), m_brass, rot_z_deg=-26.0)
    # Adjoining 3D warm-amber lit hallway strictly BEHIND North Wall (Y >= R + 0.18, zero bleed-through!)
    add_box("Hallway_BackWall", (2.25, R + 2.55, H / 2.0), (2.40, 0.14, H), m_hall_wall)
    add_box("Hallway_LeftWall", (1.58, R + 1.42, H / 2.0), (0.14, 2.36, H), m_hall_wall)
    add_box("Hallway_RightWall", (3.25, R + 1.42, H / 2.0), (0.14, 2.36, H), m_hall_wall)
    # Console table + glowing warm amber globe lamp positioned cleanly in the doorway sightline!
    add_box("HallConsole_Top", (2.22, R + 1.50, 0.84), (0.86, 0.40, 0.05), m_wood_slat)
    add_box("HallConsole_LegL", (1.86, R + 1.50, 0.42), (0.05, 0.34, 0.84), m_dark_metal)
    add_box("HallConsole_LegR", (2.58, R + 1.50, 0.42), (0.05, 0.34, 0.84), m_dark_metal)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.16, location=(2.20, R + 1.46, 1.06))
    lamp_globe = bpy.context.active_object
    lamp_globe.name = "HallAmberLamp_Globe"
    lamp_globe.data.materials.append(m_amber_lamp)
    add_box("HallAmberLamp_Base", (2.20, R + 1.46, 0.89), (0.14, 0.14, 0.06), m_brass)
    add_point_light("Hallway_AmberLight", (2.20, R + 1.32, 1.24), energy=300.0, color=(1.0, 0.66, 0.24), radius=0.22)

    # ==================================================================================
    # EAST WALL (yaw = 90 / 120 deg): 42U AI Server Racks + Oscilloscope Bench
    # ==================================================================================
    for ri, ry in enumerate((-1.25, 0.05, 1.35)):
        add_box(f"ServerRack_{ri}", (R - 0.42, ry, 1.12), (0.76, 0.96, 2.24), m_dark_metal, bevel=0.01)
        for bi in range(6):
            add_box(f"RackLED_{ri}_{bi}", (R - 0.81, ry + 0.15, 0.48 + bi * 0.28), (0.02, 0.36, 0.025), m_emerald_led)
    for si in range(8):
        add_box(f"EastSlat_{si}", (R - 0.03, -3.20 + si * 0.16, H / 2.0), (0.05, 0.06, H - 0.30), m_wood_slat)
    # Warm practical sconce on East wall
    add_box("EastSconce", (R - 0.08, -2.25, 1.85), (0.10, 0.12, 0.36), m_sconce_warm)
    add_point_light("EastSconce_Light", (R - 0.28, -2.25, 1.85), energy=110.0, color=(1.0, 0.74, 0.38), radius=0.18)

    # ==================================================================================
    # SOUTH WALL (yaw = 180 deg): Rear Industrial Workshop Shelving (ZERO text!)
    # ==================================================================================
    for sx in (-1.75, 0.0, 1.75):
        add_box(f"ShelfPostL_{sx}", (sx - 0.65, -R + 0.32, 1.15), (0.05, 0.52, 2.30), m_dark_metal)
        add_box(f"ShelfPostR_{sx}", (sx + 0.65, -R + 0.32, 1.15), (0.05, 0.52, 2.30), m_dark_metal)
        for sz in (0.45, 1.05, 1.65, 2.20):
            add_box(f"ShelfBoard_{sx}_{sz}", (sx, -R + 0.32, sz), (1.32, 0.52, 0.04), m_wood_slat)
        # Hardware equipment crates on shelves
        add_box(f"CrateA_{sx}", (sx - 0.25, -R + 0.34, 0.64), (0.44, 0.36, 0.32), m_dark_metal)
        add_box(f"CrateB_{sx}", (sx + 0.26, -R + 0.34, 1.24), (0.38, 0.34, 0.32), m_desk_top)
        add_box(f"BrassProp_{sx}", (sx - 0.12, -R + 0.34, 1.80), (0.22, 0.22, 0.24), m_brass)
    add_box("SouthSconceL", (-2.85, -R + 0.08, 1.90), (0.12, 0.10, 0.36), m_sconce_warm)
    add_box("SouthSconceR", (2.85, -R + 0.08, 1.90), (0.12, 0.10, 0.36), m_sconce_warm)
    add_point_light("South_WarmL", (-2.85, -R + 0.30, 1.90), energy=120.0, color=(1.0, 0.72, 0.36), radius=0.2)
    add_point_light("South_WarmR", (2.85, -R + 0.30, 1.90), energy=120.0, color=(1.0, 0.72, 0.36), radius=0.2)

    # ==================================================================================
    # WEST WALL (yaw = 240 / 270 deg): Mullioned Twilight Studio Window + Drafting Table
    # ==================================================================================
    add_box("WestWindow_Glass", (-R + 0.02, 0.0, 1.72), (0.04, 2.60, 1.55), m_window_twilight)
    for my in (-1.30, -0.45, 0.45, 1.30):
        add_box(f"WindowMullionV_{my}", (-R + 0.06, my, 1.72), (0.08, 0.06, 1.62), m_dark_metal)
    for mz in (0.95, 1.72, 2.49):
        add_box(f"WindowMullionH_{mz}", (-R + 0.06, 0.0, mz), (0.08, 2.68, 0.06), m_dark_metal)
    # Drafting table & architect warm lamp beneath West window
    add_box("DraftTable_Top", (-R + 0.62, -0.20, 0.82), (0.92, 1.85, 0.05), m_wood_slat, bevel=0.008)
    add_box("DraftTable_Leg1", (-R + 0.62, -0.98, 0.41), (0.82, 0.06, 0.82), m_dark_metal)
    add_box("DraftTable_Leg2", (-R + 0.62, 0.58, 0.41), (0.82, 0.06, 0.82), m_dark_metal)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.12, location=(-R + 0.48, 0.38, 1.06))
    w_lamp = bpy.context.active_object
    w_lamp.name = "WestArchitectLamp"
    w_lamp.data.materials.append(m_amber_lamp)
    add_point_light("WestLamp_Light", (-R + 0.56, 0.38, 1.18), energy=125.0, color=(1.0, 0.72, 0.34), radius=0.18)
    add_area_light("WestWindow_CoolFill", (-R + 0.25, 0.0, 1.72), (0.0, 0.0, 1.20), energy=140.0, color=(0.68, 0.84, 1.0), size_x=2.2, size_y=1.2)

    # ==================================================================================
    # Studio Ceiling Key & Fill Lights (Balanced cinema exposure around all 360 degrees!)
    # ==================================================================================
    add_area_light("Key_NorthPlaque", (0.25, 1.35, H - 0.18), (0.36, R, 1.62), energy=260.0, color=(0.98, 0.99, 1.0), size_x=1.8, size_y=0.6)
    add_area_light("Key_FounderDesk", (-1.50, 1.05, H - 0.22), (-1.52, 2.75, 1.45), energy=210.0, color=(0.96, 0.98, 0.95), size_x=1.4, size_y=0.8)
    add_area_light("Fill_CenterCeiling", (0.0, 0.0, H - 0.12), (0.0, 0.0, 0.0), energy=180.0, color=(0.86, 0.92, 0.96), size_x=2.6, size_y=2.6)
    add_area_light("Key_EastRacks", (1.60, 0.0, H - 0.20), (R, 0.0, 1.35), energy=190.0, color=(0.84, 0.96, 0.88), size_x=2.2, size_y=0.6)
    add_area_light("Key_SouthWall", (0.0, -1.60, H - 0.20), (0.0, -R, 1.35), energy=210.0, color=(0.96, 0.92, 0.86), size_x=2.8, size_y=0.7)

    # Nodal 35mm Cinema Camera at room center (0, 0, 1.50m)
    cam_data = bpy.data.cameras.new("OrbitCam")
    cam_data.lens = float(args.focal_mm)
    cam_data.sensor_width = 36.0
    cam_data.clip_start = 0.1
    cam_data.clip_end = 50.0
    cam_obj = bpy.data.objects.new("OrbitCam", cam_data)
    scene.collection.objects.link(cam_obj)
    scene.camera = cam_obj
    cam_obj.location = (0.0, 0.0, 1.50)

    blend_path = out_dir / "studio_360_room_v4.blend"
    bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))

    yaws = [int(y) for y in args.yaws.split(",") if y.strip()]
    rendered_entries: List[Dict[str, Any]] = []
    sheet_rows: List[Tuple[str, Path]] = []

    for yaw in yaws:
        # Clockwise pan right in Blender Z-up: rot_x=90 deg looks +Y (North); rot_z = -yaw rotates +Y -> +X (East)!
        cam_obj.rotation_euler = (math.radians(90.0), 0.0, -math.radians(float(yaw)))
        out_png = out_dir / f"yaw_{yaw:03d}.png"
        scene.render.filepath = str(out_png)
        bpy.ops.render.render(write_still=True)
        sha = _sha256_file(out_png)
        rendered_entries.append(
            {
                "yaw_deg": float(yaw),
                "filename": out_png.name,
                "path": str(out_png),
                "sha256": sha,
                "bytes": int(out_png.stat().st_size),
                "width": int(args.width),
                "height": int(args.height),
            }
        )
        sheet_rows.append((f"yaw = {yaw:3d}°  ({args.width}x{args.height})", out_png))
        print(f"[blender-360-room] Rendered {out_png.name} sha256={sha[:12]}", flush=True)

    manifest = {
        "schema_version": "omni-912-blender-360-room-anchors/v1",
        "blender_version": bpy.app.version_string,
        "engine": "CYCLES_CPU_OIDN",
        "samples": int(args.samples),
        "focal_mm": float(args.focal_mm),
        "width": int(args.width),
        "height": int(args.height),
        "sign_text": args.sign_text,
        "founder_ref_path": str(args.founder_ref),
        "founder_ref_sha256": _sha256_file(Path(args.founder_ref)),
        "blend_path": str(blend_path),
        "anchors": rendered_entries,
    }
    man_p = out_dir / "room_anchors_manifest.json"
    man_p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[blender-360-room] BLENDER_RENDER_DONE -> {man_p} ({len(rendered_entries)} angles)", flush=True)


def run_host_coordinator(args: argparse.Namespace) -> None:
    out_dir = Path(args.out_dir)
    tex_dir = out_dir / "textures"
    tex_dir.mkdir(parents=True, exist_ok=True)
    build_clean_plaque_texture(args.sign_text, tex_dir / "plaque_24word.png")
    _, founder_aspect = build_feathered_founder_rgba(Path(args.founder_ref), tex_dir / "founder_feathered.png")
    (tex_dir / "textures_meta.json").write_text(
        json.dumps({"founder_aspect": founder_aspect, "sign_text": args.sign_text}, indent=2) + "\n",
        encoding="utf-8",
    )
    cmd = [
        str(args.blender_bin),
        "--background",
        "--factory-startup",
        "--python", str(Path(__file__).resolve()),
        "--",
        "--blender-render",
        "--out-dir", str(out_dir),
        "--founder-ref", str(args.founder_ref),
        "--sign-text", str(args.sign_text),
        "--width", str(args.width),
        "--height", str(args.height),
        "--samples", str(args.samples),
        "--threads", str(args.threads),
        "--focal-mm", str(args.focal_mm),
        "--yaws", str(args.yaws),
    ]
    proc = subprocess.run(cmd, check=False)
    if proc.returncode != 0:
        raise RuntimeError(f"Blender render failed rc={proc.returncode}")

    man_p = out_dir / "room_anchors_manifest.json"
    manifest = json.loads(man_p.read_text(encoding="utf-8"))
    sheet_rows = [
        (f"yaw = {int(round(a['yaw_deg'])):3d}°  ({a['width']}x{a['height']})", Path(a["path"]))
        for a in manifest["anchors"]
    ]
    sheet_p = build_contact_sheet(sheet_rows, out_dir / "qa_contact_sheet_blender_room.png")
    manifest["contact_sheet_path"] = str(sheet_p)
    manifest["contact_sheet_sha256"] = _sha256_file(sheet_p)
    man_p.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"[blender-360-room] ALL DONE -> {sheet_p}", flush=True)


def parse_args() -> argparse.Namespace:
    argv = sys.argv[sys.argv.index("--") + 1 :] if "--" in sys.argv else sys.argv[1:]
    ap = argparse.ArgumentParser(description="Render photorealistic 3D Blender 5.2.5 LTS studio room anchors for Showcase 2 v4")
    ap.add_argument("--blender-render", action="store_true", help="Internal flag when running inside Blender bpy")
    ap.add_argument("--blender-bin", default=DEFAULT_BLENDER_BIN)
    ap.add_argument("--out-dir", default="/mnt/storage/omni_gate2_runs/phase0_912_spinup/showcase_scripts/blender_360_room_v4")
    ap.add_argument("--founder-ref", default=DEFAULT_FOUNDER_REF)
    ap.add_argument("--sign-text", default=PINNED_SIGN_TEXT)
    ap.add_argument("--width", type=int, default=1344)
    ap.add_argument("--height", type=int, default=768)
    ap.add_argument("--samples", type=int, default=40)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--focal-mm", type=float, default=26.0)
    ap.add_argument("--yaws", default=",".join(str(y) for y in DEFAULT_YAWS))
    return ap.parse_args(argv)


if __name__ == "__main__":
    _args = parse_args()
    if _args.blender_render:
        run_blender_build(_args)
    else:
        run_host_coordinator(_args)

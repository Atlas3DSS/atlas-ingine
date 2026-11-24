import bpy
import math
from mathutils import Vector
from math import radians

# ---------- RESET ----------
bpy.ops.wm.read_factory_settings(use_empty=True)

# ---------- SCENE / OUTPUT ----------
scene = bpy.context.scene
scene.render.resolution_x = 1920
scene.render.resolution_y = 1080
scene.render.resolution_percentage = 100
scene.render.fps = 24
scene.frame_start = 1
scene.frame_end = 240
scene.render.image_settings.file_format = 'PNG'
scene.render.film_transparent = False
scene.view_settings.view_transform = 'AgX'
scene.view_settings.exposure = 1.05
scene.world = bpy.data.worlds.new('World')
scene.world.use_nodes = True
wnodes = scene.world.node_tree.nodes
wnodes.get('Background').inputs['Color'].default_value = (0.03, 0.04, 0.05, 1.0)
wnodes.get('Background').inputs['Strength'].default_value = 0.55

# ---------- PALETTE ----------
CHARCOAL = (0.063, 0.082, 0.094, 1.0)
BLUE_STEEL = (0.333, 0.427, 0.475, 1.0)
DARK_GREY = (0.408, 0.408, 0.408, 1.0)
EMERALD = (0.510, 0.761, 0.196, 1.0)
LIGHT_GREY = (0.878, 0.878, 0.878, 1.0)
WHITE = (1.0, 1.0, 1.0, 1.0)

# ---------- HELPERS ----------
def noise_rough(mat, scale=18.0, strength=0.35, base=0.35):
    nt = mat.node_tree
    bsdf = nt.nodes.get('Principled BSDF')
    nz = nt.nodes.new('ShaderNodeTexNoise')
    nz.inputs['Scale'].default_value = scale
    nz.inputs['Detail'].default_value = 2.0
    cr = nt.nodes.new('ShaderNodeValToRGB')
    cr.color_ramp.elements[0].position = 0.0
    cr.color_ramp.elements[0].color = (base, base, base, 1.0)
    cr.color_ramp.elements[1].position = 1.0
    cr.color_ramp.elements[1].color = (min(base+strength,1.0),)*3 + (1.0,)
    nt.links.new(nz.outputs['Fac'], cr.inputs['Fac'])
    nt.links.new(cr.outputs['Color'], bsdf.inputs['Roughness'])

def make_mat(name, color, metallic=0.85, rough=0.35, emission=None, emis_str=0.0, transmission=0.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    bsdf = nt.nodes['Principled BSDF']
    bsdf.inputs['Base Color'].default_value = color
    bsdf.inputs['Metallic'].default_value = metallic
    bsdf.inputs['Roughness'].default_value = rough
    if emission is not None:
        bsdf.inputs['Emission Color'].default_value = emission
        bsdf.inputs['Emission Strength'].default_value = emis_str
    if transmission > 0.0:
        bsdf.inputs['Transmission Weight'].default_value = transmission
        bsdf.inputs['IOR'].default_value = 1.45
        bsdf.inputs['Roughness'].default_value = 0.08
        bsdf.inputs['Metallic'].default_value = 0.0
    return m

def add_box(name, loc, scale, mat, bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1.0, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.scale = scale
    o.data.materials.append(mat)
    if bevel > 0.0:
        mod = o.modifiers.new('Bevel', 'BEVEL')
        mod.width = bevel
        mod.segments = 3
        mod.limit_method = 'ANGLE'
        mod.angle_limit = radians(40)
    return o

def add_cyl(name, loc, radius, depth, mat, verts=48, bevel=0.0):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.data.materials.append(mat)
    if bevel > 0.0:
        mod = o.modifiers.new('Bevel', 'BEVEL')
        mod.width = bevel
        mod.segments = 3
    return o

# ---------- MATERIALS ----------
mat_floor = make_mat('ObsidianFloor', CHARCOAL, metallic=0.55, rough=0.28)
noise_rough(mat_floor, scale=30.0, strength=0.18, base=0.22)
mat_steel = make_mat('BlueSteel', BLUE_STEEL, metallic=0.9, rough=0.32)
noise_rough(mat_steel, scale=22.0, strength=0.22, base=0.25)
mat_grey = make_mat('DarkGrey', DARK_GREY, metallic=0.6, rough=0.45)
mat_emerald_glow = make_mat('EmeraldGlow', (0.05,0.10,0.04,1.0), metallic=0.2, rough=0.4,
                            emission=EMERALD, emis_str=3.2)
mat_white_glow = make_mat('WhiteGlow', (0.9,0.95,1.0,1.0), metallic=0.0, rough=0.4,
                          emission=WHITE, emis_str=4.5)
mat_glass = make_mat('CrystalGlass', (0.75,0.85,0.9,1.0), metallic=0.0, rough=0.08,
                     transmission=0.85)
mat_light = make_mat('LightGrey', LIGHT_GREY, metallic=0.4, rough=0.4)

# ---------- FLOOR ----------
floor = add_box('Floor', (0,0,-0.05), (18,18,0.1), mat_floor)

# ---------- MONOLITHS (pushed to edges) ----------
years = ['2016','2017','2018','2020','2022','2026']
names = ['Pascal','Volta','Turing','Ampere','Hopper','Blackwell']
heights = [1.6, 2.1, 2.6, 3.1, 3.6, 4.2]
widths = [1.05, 1.05, 1.0, 1.0, 0.95, 0.9]
depths = [0.55, 0.55, 0.5, 0.5, 0.45, 0.42]
n = len(years)
monoliths = []
for i in range(n):
    t = i/(n-1)
    x = -7.6 + t*15.2
    y = -2.6 + math.sin(t*math.pi)*0.9
    h = heights[i]; w = widths[i]; d = depths[i]
    plinth = add_box(f'Plinth_{years[i]}', (x,y,0.12), (w+0.55, d+0.55, 0.24), mat_grey, bevel=0.05)
    body = add_box(f'Monolith_{years[i]}', (x,y,0.24+h/2), (w, d, h), mat_steel, bevel=0.06)
    cap = add_box(f'Cap_{years[i]}', (x,y,0.24+h+0.05), (w*0.86, d*0.86, 0.10), mat_light, bevel=0.03)
    glass = add_box(f'Glass_{years[i]}', (x, y+d/2+0.02, 0.24+h*0.55), (w*0.7, 0.03, h*0.55), mat_glass)
    trace_count = 3 + i
    for j in range(trace_count):
        ty = 0.45 + (h-0.7) * (j/max(trace_count-1,1))
        tr = add_box(f'Trace_{years[i]}_{j}', (x, y+d/2+0.015, 0.24+ty), (w*0.72, 0.02, 0.035), mat_emerald_glow)
    sp = add_box(f'Spine_{years[i]}', (x, y+d/2+0.015, 0.24+h/2), (0.045, 0.02, h*0.85), mat_emerald_glow)
    monoliths.append({'year':years[i],'name':names[i],'x':x,'y':y,'h':h})

# ---------- 3D YEAR MARKERS ----------
def add_text(text, loc, size=0.55, mat=None, rotation=(0,0,0)):
    bpy.ops.object.text_add(location=loc, rotation=rotation)
    o = bpy.context.active_object
    o.name = 'Text_'+text.replace(' ','_')
    o.data.body = text
    o.data.align_x = 'CENTER'
    o.data.align_y = 'CENTER'
    try:
        fpath = '/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf'
        o.data.font = bpy.data.fonts.load(fpath)
    except Exception:
        pass
    o.data.size = size
    o.data.extrude = 0.05
    if mat: o.data.materials.append(mat)
    return o

for m in monoliths:
    add_text(m['year'], (m['x'], m['y']-1.15, 0.22), size=0.42, mat=mat_light, rotation=(0,0,0))

# ---------- HERO PEDESTAL + CORE ----------
hero_x, hero_y = 0.0, 2.6
pedestal = add_box('HeroPedestal', (hero_x, hero_y, 0.45), (2.4, 2.4, 0.9), mat_grey, bevel=0.08)
pedestal_cap = add_box('HeroPedestalCap', (hero_x, hero_y, 0.95), (1.9, 1.9, 0.12), mat_light, bevel=0.04)
pedestal_glow = add_box('HeroPedestalGlow', (hero_x, hero_y, 1.03), (1.5, 1.5, 0.04), mat_emerald_glow)

core_group = bpy.data.objects.new('InklingCore', None)
scene.collection.objects.link(core_group)
core_group.location = Vector((hero_x, hero_y, 3.05))

# Flat stacked rings (no AABB overlap with crystal at center)
ring_specs = [
    (1.30, 0.14, -0.85, 'BLUE_STEEL'),
    (1.00, 0.12, +0.85, 'EMERALD'),
    (0.62, 0.10, +1.55, 'WHITE'),
]
rings = []
for idx,(r,tube,z,kind) in enumerate(ring_specs):
    bpy.ops.mesh.primitive_torus_add(major_radius=r, minor_radius=tube, major_segments=72, minor_segments=18, location=(0,0,0))
    o = bpy.context.active_object
    o.name = f'CoreRing_{idx}'
    o.parent = core_group
    o.location = (0,0,z)
    if kind == 'BLUE_STEEL':
        o.data.materials.append(mat_steel)
    elif kind == 'EMERALD':
        o.data.materials.append(mat_emerald_glow)
    else:
        o.data.materials.append(mat_white_glow)
    o.rotation_euler = (0, 0, radians(idx*20))
    rings.append(o)

# Crystal (glass) at center — no nested inner sphere
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.46, subdivisions=2, location=(0,0,0))
core_crystal = bpy.context.active_object
core_crystal.name = 'CoreCrystal'
core_crystal.parent = core_group
core_crystal.location = (0,0,0)
core_crystal.data.materials.append(mat_glass)

# emerald light-bridges from Blackwell monolith to core
blackwell = monoliths[-1]
bridge_start = Vector((blackwell['x'], blackwell['y'], blackwell['h']+0.5))
bridge_end = Vector((hero_x, hero_y, 3.0))
for j in range(3):
    for k in range(1, 9):
        t = k/9.0
        p = bridge_start.lerp(bridge_end, t)
        p.z += math.sin(t*math.pi)*(0.9 + j*0.25)
        bpy.ops.mesh.primitive_uv_sphere_add(radius=0.05, location=p)
        seg = bpy.context.active_object
        seg.name = f'BridgeSeg_{j}_{k}'
        seg.data.materials.append(mat_emerald_glow)

# ---------- GEOMETRY NODES ORBITAL CONSTELLATION ----------
bpy.ops.mesh.primitive_ico_sphere_add(radius=0.001, location=(hero_x, hero_y, 3.05))
gn_host = bpy.context.active_object
gn_host.name = 'OrbitalConstellation'

mod = gn_host.modifiers.new('OrbitalGN', 'NODES')
ng = bpy.data.node_groups.new('OrbitalTree', 'GeometryNodeTree')
mod.node_group = ng
ng.interface.new_socket(name='Geometry', in_out='INPUT', socket_type='NodeSocketGeometry')
ng.interface.new_socket(name='Geometry', in_out='OUTPUT', socket_type='NodeSocketGeometry')
ni = ng.nodes.new('NodeGroupInput')
no = ng.nodes.new('NodeGroupOutput')
ico = ng.nodes.new('GeometryNodeMeshIcoSphere')
ico.inputs['Radius'].default_value = 2.35
ico.inputs['Subdivisions'].default_value = 3
sph = ng.nodes.new('GeometryNodeMeshIcoSphere')
sph.inputs['Radius'].default_value = 0.06
sph.inputs['Subdivisions'].default_value = 1
inst = ng.nodes.new('GeometryNodeInstanceOnPoints')
ng.links.new(ico.outputs['Mesh'], inst.inputs['Points'])
ng.links.new(sph.outputs['Mesh'], inst.inputs['Instance'])
ng.links.new(inst.outputs['Instances'], no.inputs['Geometry'])
gn_host.data.materials.append(mat_emerald_glow)

# ---------- LIGHTING ----------
def add_light(name, loc, energy, color, type='AREA', size=4.0):
    bpy.ops.object.light_add(type=type, location=loc)
    o = bpy.context.active_object
    o.name = name
    o.data.energy = energy
    o.data.color = color
    if type == 'AREA':
        o.data.size = size
    target = bpy.data.objects.new(name+'_Target', None)
    scene.collection.objects.link(target)
    target.location = (0, 1.2, 2.0)
    tgt_con = o.constraints.new('TRACK_TO')
    tgt_con.target = target
    tgt_con.track_axis = 'TRACK_NEGATIVE_Z'
    tgt_con.up_axis = 'UP_Y'
    return o

key = add_light('KeyLight', (3.5, -6.5, 9.0), 950, (0.86, 0.93, 1.0), 'AREA', 5.0)
rim_green = add_light('GreenKick', (-7.5, 3.5, 5.0), 700, (0.51, 0.76, 0.20), 'AREA', 3.5)
fill = add_light('BlueSteelFill', (7.5, 2.0, 3.5), 420, (0.42, 0.55, 0.62), 'AREA', 4.0)
back = add_light('CoreBacklight', (0.0, 7.5, 5.0), 850, (0.95, 1.0, 0.95), 'AREA', 3.0)
hero_spot = add_light('HeroSpot', (hero_x, hero_y-4.0, 7.0), 1200, (1.0, 0.95, 0.85), 'SPOT', 2.0)
hero_spot.data.spot_size = radians(45)
hero_spot.data.spot_blend = 0.6

# ---------- CAMERA (wide, elevated, hero-centered) ----------
bpy.ops.object.empty_add(type='PLAIN_AXES', location=(0, 2.2, 2.9))
tgt = bpy.context.active_object
tgt.name = 'CameraTarget'
bpy.ops.object.camera_add(location=(-8.5, -10.5, 6.0))
cam = bpy.context.active_object
cam.name = 'HeroCamera'
scene.camera = cam
cam.data.lens = 40
cam.data.clip_start = 0.05
cam.data.clip_end = 100
tr = cam.constraints.new('TRACK_TO')
tr.target = tgt
tr.track_axis = 'TRACK_NEGATIVE_Z'
tr.up_axis = 'UP_Y'

cam_keys = [
    (1,   (-8.5, -10.5, 6.0), (-0.5, 1.6, 2.7)),
    (60,  (-5.5, -9.0, 5.4),  (-0.3, 2.0, 2.8)),
    (120, (0.0, -8.8, 5.0),   (0.0, 2.3, 2.9)),
    (180, (5.5, -8.6, 5.2),   (0.3, 2.4, 2.9)),
    (240, (2.5, -8.0, 4.6),   (0.0, 2.6, 3.0)),
]
for fr, loc, tloc in cam_keys:
    cam.location = Vector(loc)
    cam.keyframe_insert('location', frame=fr)
    tgt.location = Vector(tloc)
    tgt.keyframe_insert('location', frame=fr)

for fc in cam.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.handle_left_type = 'AUTO'
        kp.handle_right_type = 'AUTO'
        kp.interpolation = 'BEZIER'
for fc in tgt.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.handle_left_type = 'AUTO'
        kp.handle_right_type = 'AUTO'
        kp.interpolation = 'BEZIER'

# ---------- CORE ANIMATION ----------
core_group.location = Vector((hero_x, hero_y, 3.05))
core_group.rotation_euler = (0,0,0)
core_group.keyframe_insert('rotation_euler', frame=1, index=2)
core_group.keyframe_insert('location', frame=1)
core_group.rotation_euler = (0,0,0.9)
core_group.location = Vector((hero_x, hero_y, 3.35))
core_group.keyframe_insert('rotation_euler', frame=240, index=2)
core_group.keyframe_insert('location', frame=240)
core_group.location = Vector((hero_x, hero_y, 3.05))
core_group.keyframe_insert('location', frame=120)
for fc in core_group.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'
        kp.handle_left_type = 'AUTO'
        kp.handle_right_type = 'AUTO'

for i, r in enumerate(rings):
    z = [-0.85, 0.85, 1.55][i]
    r.location = (0,0,z)
    r.keyframe_insert('rotation_euler', frame=1, index=2)
    r.rotation_euler.z += 0.7
    r.keyframe_insert('rotation_euler', frame=240, index=2)
    for fc in r.animation_data.action.fcurves:
        for kp in fc.keyframe_points:
            kp.interpolation = 'BEZIER'
            kp.handle_left_type = 'AUTO'
            kp.handle_right_type = 'AUTO'

gn_host.rotation_euler.z = 0.0
gn_host.keyframe_insert('rotation_euler', frame=1, index=2)
gn_host.rotation_euler.z = 1.4
gn_host.keyframe_insert('rotation_euler', frame=240, index=2)
for fc in gn_host.animation_data.action.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'
        kp.handle_left_type = 'AUTO'
        kp.handle_right_type = 'AUTO'

# ---------- COMPOSITOR ----------
scene.use_nodes = True
cnt = scene.node_tree
cnt.nodes.clear()
rl = cnt.nodes.new('CompositorNodeRLayers')
gl = cnt.nodes.new('CompositorNodeGlare')
gl.glare_type = 'FOG_GLOW'
gl.threshold = 1.0
gl.size = 7
co = cnt.nodes.new('CompositorNodeComposite')
cnt.links.new(rl.outputs['Image'], gl.inputs['Image'])
cnt.links.new(gl.outputs['Image'], co.inputs['Image'])

# ---------- FINAL ----------
bpy.context.view_layer.update()
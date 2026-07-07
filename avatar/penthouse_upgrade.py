"""Idempotent penthouse detail pass.

Run after apartment.py/furnish.py/apt_polish.py and before export_apartment.py.
Adds cove lighting, balcony glass, floor inlay, city shimmer, and richer decor.
"""
import bpy, math, random

PREFIX = "upg_"
random.seed(23)

col = bpy.data.collections["Apartment"]

for ob in list(bpy.data.objects):
    if ob.name.startswith(PREFIX):
        bpy.data.objects.remove(ob, do_unlink=True)

def mat(name, color, rough=0.5, metal=0.0, emit=None, emit_str=1.0, alpha=None):
    name = PREFIX + name
    m = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    m.use_nodes = True
    bsdf = m.node_tree.nodes.get("Principled BSDF")
    if bsdf:
        bsdf.inputs["Base Color"].default_value = (*color, 1)
        bsdf.inputs["Roughness"].default_value = rough
        bsdf.inputs["Metallic"].default_value = metal
        if emit:
            bsdf.inputs["Emission Color"].default_value = (*emit, 1)
            bsdf.inputs["Emission Strength"].default_value = emit_str
        if alpha is not None:
            bsdf.inputs["Alpha"].default_value = alpha
            m.blend_method = 'BLEND'
            if hasattr(m, "use_screen_refraction"):
                m.use_screen_refraction = True
    return m

M_gold = bpy.data.materials.get("GoldTrim") or mat("gold", (1.0, 0.72, 0.25), 0.22, 1.0)
M_glass = mat("smoked_glass", (0.62, 0.78, 0.95), 0.02, alpha=0.18)
M_black = mat("blackened_steel", (0.02, 0.02, 0.025), 0.52, 0.2)
M_warm = mat("warm_cove", (1.0, 0.78, 0.44), 0.3, emit=(1.0, 0.72, 0.36), emit_str=7.5)
M_cool = mat("cool_city", (0.48, 0.66, 1.0), 0.4, emit=(0.34, 0.54, 1.0), emit_str=2.8)
M_fire = mat("linear_fire", (1.0, 0.38, 0.10), 0.4, emit=(1.0, 0.32, 0.06), emit_str=9.0)
M_leaf = mat("deep_leaf", (0.035, 0.18, 0.07), 0.75)
M_stone = mat("warm_stone", (0.44, 0.39, 0.32), 0.82)

def link(ob):
    for c in list(ob.users_collection):
        c.objects.unlink(ob)
    col.objects.link(ob)
    return ob

def box(name, size, loc, material, bevel=0.0, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = PREFIX + name
    ob.scale = (size[0] / 2, size[1] / 2, size[2] / 2)
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)
    if bevel:
        bv = ob.modifiers.new("soft_edges", "BEVEL")
        bv.width = bevel
        bv.segments = 4
    ob.data.materials.append(material)
    return link(ob)

def cyl(name, radius, depth, loc, material, verts=32, rot=(0, 0, 0)):
    bpy.ops.mesh.primitive_cylinder_add(vertices=verts, radius=radius, depth=depth, location=loc, rotation=rot)
    ob = bpy.context.active_object
    ob.name = PREFIX + name
    ob.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return link(ob)

def sph(name, radius, loc, material, scale=(1, 1, 1)):
    bpy.ops.mesh.primitive_uv_sphere_add(segments=24, ring_count=12, radius=radius, location=loc)
    ob = bpy.context.active_object
    ob.name = PREFIX + name
    ob.scale = scale
    ob.data.materials.append(material)
    bpy.ops.object.shade_smooth()
    return link(ob)

def light(name, kind, loc, energy, color, size=1.0):
    data = bpy.data.lights.new(PREFIX + name, kind)
    data.energy = energy
    data.color = color
    if kind == 'AREA':
        data.size = size
    ob = bpy.data.objects.new(PREFIX + name, data)
    ob.location = loc
    col.objects.link(ob)
    return ob

# Architecture and lighting.
box("balcony_glass", (7.8, 0.035, 0.72), (0, -3.18, 0.66), M_glass, bevel=0.01)
box("balcony_top_rail", (7.9, 0.05, 0.06), (0, -3.12, 1.04), M_gold, bevel=0.01)
box("balcony_floor_glow", (7.6, 0.035, 0.028), (0, -3.04, 0.08), M_warm)
box("ceiling_cove_front", (7.4, 0.10, 0.028), (0, -2.5, 2.93), M_warm)
box("ceiling_cove_lounge", (5.6, 0.10, 0.028), (-0.2, 2.25, 2.93), M_warm)
box("gallery_wash_left", (0.028, 2.4, 0.035), (-4.38, -0.35, 2.48), M_cool)
light("soft_front_cove", 'AREA', (0, -2.2, 2.72), 120, (1.0, 0.78, 0.48), 4.2)
light("gallery_cool_bounce", 'AREA', (-3.8, -0.25, 2.1), 70, (0.52, 0.66, 1.0), 2.2)

# Lounge focal details.
box("linear_firebox", (1.7, 0.055, 0.34), (0, 3.37, 0.72), M_black, bevel=0.012)
box("linear_fire", (1.48, 0.025, 0.075), (0, 3.325, 0.72), M_fire, bevel=0.01)
box("fire_stone_cap", (1.86, 0.08, 0.06), (0, 3.30, 0.93), M_stone, bevel=0.012)
light("fire_flicker", 'POINT', (0, 3.05, 0.82), 42, (1.0, 0.36, 0.12))

bpy.ops.mesh.primitive_torus_add(major_radius=0.92, minor_radius=0.012, major_segments=120,
                                minor_segments=8, location=(0.15, 0.25, 0.034))
ring = bpy.context.active_object
ring.name = PREFIX + "conversation_floor_inlay"
ring.data.materials.append(M_gold)
link(ring)

for x, y, s in ((-2.8, -2.85, 1.1), (3.72, -2.65, 0.86), (3.82, 3.0, 0.9)):
    cyl(f"planter_{x}_{y}", 0.18 * s, 0.28 * s, (x, y, 0.14 * s), M_black, verts=20)
    for i in range(7):
        a = i * 2.39996
        sph(f"plant_leaf_{x}_{y}_{i}", 0.14 * s,
            (x + math.cos(a) * 0.08 * s, y + math.sin(a) * 0.08 * s, (0.42 + i * 0.075) * s),
            M_leaf, scale=(0.65, 0.35, 1.8))

# Extra parallax windows behind the generated skyline texture.
window_mats = [
    mat("city_gold", (1.0, 0.78, 0.36), 0.4, emit=(1.0, 0.72, 0.36), emit_str=2.2),
    mat("city_blue", (0.45, 0.62, 1.0), 0.4, emit=(0.36, 0.52, 1.0), emit_str=1.7),
    mat("city_white", (0.96, 0.95, 0.88), 0.4, emit=(0.95, 0.90, 0.76), emit_str=1.5),
]
for i in range(120):
    x = random.uniform(-9.6, 9.6)
    z = random.uniform(0.55, 6.4)
    y = random.uniform(-7.6, -6.6)
    box(f"city_pin_{i:03d}", (random.uniform(0.035, 0.09), 0.018, random.uniform(0.045, 0.13)),
        (x, y, z), random.choice(window_mats))

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("PENTHOUSE_UPGRADE_DONE")

"""Deep furnishing: kitchen island + stools + pendants, bar wall, plants, bedroom zone."""
import bpy, math, random
from mathutils import Vector

random.seed(11)
col = bpy.data.collections["Apartment"]
S = bpy.context.scene

def mat(name, color, rough=0.5, metal=0.0, emit=None, emit_str=1.0, alpha=None):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_str
    if alpha is not None:
        b.inputs["Alpha"].default_value = alpha; m.blend_method = 'BLEND'
    return m

def link(ob):
    for c in ob.users_collection: c.objects.unlink(ob)
    col.objects.link(ob)
    return ob

def box(name, size, loc, m, bevel=0.0, rot=(0,0,0)):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    ob = bpy.context.active_object; ob.name = name
    ob.scale = (size[0]/2, size[1]/2, size[2]/2)
    bpy.ops.object.transform_apply(scale=True)
    if bevel: bv = ob.modifiers.new("B","BEVEL"); bv.width = bevel; bv.segments = 3
    ob.data.materials.append(m)
    return link(ob)

def cyl(name, r, depth, loc, m, verts=24):
    bpy.ops.mesh.primitive_cylinder_add(radius=r, depth=depth, location=loc, vertices=verts)
    ob = bpy.context.active_object; ob.name = name
    ob.data.materials.append(m); bpy.ops.object.shade_smooth()
    return link(ob)

def sph(name, r, loc, m, scale=(1,1,1)):
    bpy.ops.mesh.primitive_uv_sphere_add(radius=r, location=loc, segments=20, ring_count=12)
    ob = bpy.context.active_object; ob.name = name; ob.scale = scale
    ob.data.materials.append(m); bpy.ops.object.shade_smooth()
    return link(ob)

M_marble = bpy.data.materials["MarbleFloor"]
M_gold = bpy.data.materials["GoldTrim"]
M_wood = bpy.data.materials["DarkWood"]
M_black = mat("MatteBlack", (0.03,0.03,0.035), 0.6)
M_leaf = mat("Leaf", (0.06, 0.22, 0.08), 0.7)
M_pot  = mat("Pot", (0.12, 0.10, 0.09), 0.85)
M_bed  = mat("BedLinen", (0.88, 0.86, 0.82), 0.9)
M_throw= mat("Throw", (0.45, 0.28, 0.10), 0.85)
M_pill = mat("Pillow", (0.75, 0.70, 0.62), 0.9)

# ── move the big statue out of the future bedroom corner → by the TV corner ──
delta = Vector((7.0, 0.2, 0))
for ob in list(col.objects):
    if ob.type == 'MESH' and (Vector((ob.location.x, ob.location.y, 0)) - Vector((-3.4, 2.9, 0))).length < 0.6:
        if ob.name.startswith(("Cone","Torus","Sphere","Cylinder")):
            ob.location += delta

# ── kitchen island (right-front) + stools + pendants ──
box("island_base", (1.9, 0.95, 0.85), (3.0, 0.6, 0.425), M_black, bevel=0.02)
box("island_top", (2.05, 1.08, 0.06), (3.0, 0.6, 0.88), M_marble, bevel=0.01)
for i, sy in enumerate((-0.05, 0.6, 1.25)):
    cyl(f"stool_pole{i}", 0.03, 0.62, (1.85, sy, 0.31), M_gold)
    cyl(f"stool_seat{i}", 0.19, 0.06, (1.85, sy, 0.65), M_black)
for i, sx in enumerate((2.45, 3.0, 3.55)):
    cyl(f"pend_wire{i}", 0.008, 0.9, (sx, 0.6, 2.55), M_black, verts=8)
    sph(f"pend_bulb{i}", 0.09, (sx, 0.6, 2.05),
        mat(f"PendGlow{i}", (1,0.8,0.5), 0.4, emit=(1.0,0.75,0.4), emit_str=10))
    l = bpy.data.lights.new(f"pend_l{i}", 'POINT'); l.energy = 25; l.color = (1.0, 0.8, 0.55)
    lo = bpy.data.objects.new(f"pend_l{i}", l); lo.location = (sx, 0.6, 1.95); col.objects.link(lo)

# ── bar wall (right) : counter + shelves + bottles ──
box("bar_counter", (0.5, 2.0, 1.0), (4.15, 1.9, 0.5), M_wood, bevel=0.02)
box("bar_top", (0.56, 2.1, 0.05), (4.15, 1.9, 1.03), M_marble)
for si, sz in enumerate((1.6, 2.1)):
    box(f"bar_shelf{si}", (0.32, 2.0, 0.04), (4.32, 1.9, sz), M_wood)
    box(f"bar_glow{si}", (0.3, 1.96, 0.012), (4.32, 1.9, sz-0.028),
        mat(f"ShelfGlow{si}", (1,0.85,0.6), 0.5, emit=(1.0,0.8,0.5), emit_str=4))
    for bi in range(7):
        by = 1.05 + bi*0.28 + random.uniform(-0.04, 0.04)
        bh = random.uniform(0.18, 0.3)
        cc = random.choice([(0.5,0.25,0.08),(0.15,0.35,0.15),(0.4,0.35,0.1),(0.2,0.15,0.3)])
        cyl(f"bottle{si}_{bi}", 0.035, bh, (4.32, by, sz+0.02+bh/2),
            mat(f"BottleM{si}{bi}", cc, 0.1, alpha=0.75), verts=12)

# ── plants ──
def plant(name, loc, s=1.0):
    cyl(name+"_pot", 0.16*s, 0.3*s, (loc[0], loc[1], 0.15*s), M_pot)
    for i in range(5):
        a = i * 2.39996
        sph(name+f"_f{i}", 0.16*s,
            (loc[0]+math.cos(a)*0.09*s, loc[1]+math.sin(a)*0.09*s, (0.48+0.13*i)*s),
            M_leaf, scale=(1, 1, 1.9))
plant("plant1", (-4.1, -2.9), 1.15)
plant("plant2", (4.1, 3.1), 1.0)
plant("plant3", (2.3, -3.1), 0.8)

# ── bedroom zone (left) ──
box("bed_platform", (2.3, 1.9, 0.22), (-3.2, 1.5, 0.11), M_wood, bevel=0.02)
box("bed_mattress", (2.1, 1.7, 0.25), (-3.2, 1.5, 0.35), M_bed, bevel=0.06)
box("bed_throw", (2.12, 0.7, 0.27), (-3.2, 2.0, 0.36), M_throw, bevel=0.05)
box("headboard", (0.12, 1.9, 1.1), (-4.35, 1.5, 0.55), M_wood, bevel=0.02)
box("pillow1", (0.45, 0.65, 0.16), (-3.95, 1.15, 0.55), M_pill, bevel=0.06)
box("pillow2", (0.45, 0.65, 0.16), (-3.95, 1.85, 0.55), M_pill, bevel=0.06)
for i, ny in enumerate((0.35, 2.65)):
    box(f"nightstand{i}", (0.45, 0.45, 0.45), (-4.2, ny, 0.225), M_black, bevel=0.015)
    cyl(f"nlamp_base{i}", 0.03, 0.25, (-4.2, ny, 0.575), M_gold, verts=12)
    sph(f"nlamp_glow{i}", 0.09, (-4.2, ny, 0.75),
        mat(f"NLampGlow{i}", (1,0.8,0.55), 0.5, emit=(1.0,0.7,0.4), emit_str=6))
    l = bpy.data.lights.new(f"nlamp_l{i}", 'POINT'); l.energy = 18; l.color = (1.0, 0.72, 0.45)
    lo = bpy.data.objects.new(f"nlamp_l{i}", l); lo.location = (-4.05, ny, 0.85); col.objects.link(lo)
# rug under bed zone
box("bed_rug", (2.8, 2.6, 0.015), (-3.0, 1.5, 0.008), mat("BedRug", (0.22,0.19,0.16), 0.95))

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("FURNISH_DONE")

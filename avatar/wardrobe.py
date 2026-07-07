"""Wardrobe: gold bodycon dress · white crop top + jeans · sport bra + leggings.
Each garment = duplicated body faces → offset → solidify → skinned via armature.
"""
import bpy, bmesh, math
from mathutils import Vector

body = bpy.data.objects["weaver_base2-base"]
arm = bpy.data.objects["weaver_base2"]
me = body.data
gi = {g.name: g.index for g in body.vertex_groups}
def w(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]: s += g.weight
    return min(s, 1.0)

UP, LAT, DEPTH = 1, 0, 2
def mean_h(names, wmin=0.15):
    hs = [v.co[UP] for v in me.vertices if w(v, *names) > wmin]
    return sum(hs) / max(len(hs), 1)
bust_h = mean_h(["breast_L", "breast_R"])
hip_h  = mean_h(["pelvis_L", "pelvis_R"])
knee_h = mean_h(["lowerleg01_L", "lowerleg01_R"]) + 0.05
ankle_h = mean_h(["foot_L", "foot_R"]) + 0.02
torso  = abs(bust_h - hip_h)
body_d = sum(v.co[DEPTH] for v in me.vertices) / len(me.vertices)

ARMS = ("shoulder01_L","shoulder01_R","upperarm01_L","upperarm01_R","upperarm02_L",
        "upperarm02_R","lowerarm01_L","lowerarm01_R","wrist_L","wrist_R","clavicle_L","clavicle_R")

def not_arms(v, thresh=0.35): return w(v, *ARMS) < thresh

def make_garment(name, region_fn, mat, offset=0.0035, thick=0.003, relax=4, inflate_fn=None):
    gme = me.copy(); gme.name = name + "_mesh"
    ob = bpy.data.objects.new(name, gme)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = body.parent; ob.matrix_world = body.matrix_world
    am = ob.modifiers.new("Armature", "ARMATURE"); am.object = arm
    bm = bmesh.new(); bm.from_mesh(gme)
    bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
    keep = set(f.index for f in bm.faces
               if all(region_fn(gme.vertices[v.index]) for v in f.verts))
    bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context="FACES")
    bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
    bm.verts.ensure_lookup_table()
    boundary = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    for _ in range(relax):
        upd = {}
        for v in boundary:
            nbr = [e.other_vert(v) for e in v.link_edges]
            if nbr: upd[v] = sum((n.co for n in nbr), Vector()) / len(nbr)
        for v, c in upd.items(): v.co = v.co * 0.45 + c * 0.55
    bm.to_mesh(gme); bm.free()
    for v in gme.vertices:
        v.co += v.normal * offset
        if inflate_fn:
            v.co += inflate_fn(v)
    sub = ob.modifiers.new("Sub", "SUBSURF"); sub.levels = 1; sub.render_levels = 2
    sol = ob.modifiers.new("Solid", "SOLIDIFY"); sol.thickness = thick
    gme.materials.clear(); gme.materials.append(mat)
    print(f"{name}: {len(gme.polygons)} faces")
    return ob

def plain_mat(name, color, rough, metallic=0.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metallic
    return m

# ══ 1. Gold satin bodycon mini dress (strapless, mid-bust → upper thigh) ══
satin = plain_mat("DressGold", (0.85, 0.62, 0.18), 0.28, 0.45)
hem_h = hip_h - 0.42 * torso
def dress_region(v):
    h = v.co[UP]
    if not not_arms(v): return False
    if w(v, "head", "neck01", "neck02", "neck03") > 0.25: return False
    if w(v, "lowerleg01_L", "lowerleg01_R", "foot_L", "foot_R") > 0.15: return False
    top = bust_h + 0.10 * torso
    return hem_h < h < top
def dress_inflate(v):
    # flare below the hips so the hem reads as fabric, not paint
    h = v.co[UP]
    if h < hip_h:
        t = min(1.0, (hip_h - h) / (hip_h - hem_h + 1e-6))
        r = Vector((v.co[LAT], 0, v.co[DEPTH] - body_d))
        if r.length > 1e-5:
            return Vector((r.x, 0, r.z)).normalized() * (0.010 * t)
    return Vector()
dress = make_garment("outfit_dress", dress_region, satin, inflate_fn=dress_inflate)

# ══ 2. Casual: white crop top + blue skinny jeans ══
cotton = plain_mat("TopWhite", (0.92, 0.92, 0.90), 0.85)
def crop_region(v):
    h = v.co[UP]
    if not not_arms(v, 0.30): return False
    if w(v, "head", "neck01", "neck02", "neck03") > 0.25: return False
    return bust_h - 0.28 * torso < h < bust_h + 0.24 * torso
crop = make_garment("outfit_crop_top", crop_region, cotton)

denim = plain_mat("Denim", (0.16, 0.28, 0.48), 0.8)
def jeans_region(v):
    h = v.co[UP]
    if w(v, "foot_L", "foot_R") > 0.4: return False
    if h > hip_h + 0.16 * torso: return False
    if h < ankle_h + 0.02: return False
    return w(v, "pelvis_L", "pelvis_R", "spine05", "spine04", "upperleg01_L", "upperleg01_R",
             "upperleg02_L", "upperleg02_R", "lowerleg01_L", "lowerleg01_R",
             "lowerleg02_L", "lowerleg02_R") > 0.2
jeans = make_garment("outfit_jeans", jeans_region, denim, thick=0.004)

# ══ 3. Sport: bra + leggings ══
lycra_p = plain_mat("BraPurple", (0.22, 0.10, 0.35), 0.5)
def bra_region(v):
    h = v.co[UP]
    if not not_arms(v, 0.30): return False
    if w(v, "head", "neck01", "neck02", "neck03") > 0.25: return False
    return bust_h - 0.16 * torso < h < bust_h + 0.20 * torso
bra = make_garment("outfit_sport_bra", bra_region, lycra_p)

lycra_b = plain_mat("LeggingsBlack", (0.03, 0.03, 0.035), 0.45)
def legging_region(v):
    h = v.co[UP]
    if w(v, "foot_L", "foot_R") > 0.5: return False
    if h > hip_h + 0.20 * torso: return False
    return w(v, "pelvis_L", "pelvis_R", "spine05", "spine04", "upperleg01_L", "upperleg01_R",
             "upperleg02_L", "upperleg02_R", "lowerleg01_L", "lowerleg01_R",
             "lowerleg02_L", "lowerleg02_R") > 0.2
leggings = make_garment("outfit_leggings", legging_region, lycra_b)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("WARDROBE_DONE")

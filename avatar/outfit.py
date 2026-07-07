"""Black plunge-cutout monokini + gold chain strap & belly chain."""
import bpy, bmesh, math
from mathutils import Vector

body = bpy.data.objects["weaver_base2-base"]
me = body.data
gi = {g.name: g.index for g in body.vertex_groups}
def w(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]: s += g.weight
    return min(s, 1.0)

UP, LAT, DEPTH = 1, 0, 2      # mesh local axes (established earlier)
FRONT = 1.0                    # +z local = front (front_sign was +1)

# landmarks (recompute on deformed mesh)
def mean_h(names, wmin=0.15):
    hs = [v.co[UP] for v in me.vertices if w(v, *names) > wmin]
    return sum(hs) / max(len(hs), 1)
bust_h = mean_h(["breast_L", "breast_R"])
hip_h  = mean_h(["pelvis_L", "pelvis_R"])
torso  = abs(bust_h - hip_h)
crotch_h = hip_h - 0.35 * torso
body_d = sum(v.co[DEPTH] for v in me.vertices) / len(me.vertices)

def suit_region(v):
    """True if vertex belongs to the monokini fabric."""
    co = v.co
    h = co[UP]; x = co[LAT]; front = (co[DEPTH] - body_d) * FRONT > 0
    if w(v, "upperarm01_L", "upperarm01_R", "lowerarm01_L", "lowerarm01_R",
         "wrist_L", "wrist_R", "head", "neck01", "neck02", "neck03") > 0.3:
        return False
    # 1 ─ brief: pelvis band, high-cut (rises at sides)
    if crotch_h - 0.06 * torso < h < hip_h + 0.10 * torso:
        cut = (h - crotch_h) / (0.45 * torso)          # 0 at crotch → 1 at top
        max_x = 0.30 * (1.0 - 0.75 * cut)              # narrows upward = high-cut
        if abs(x) < max_x and w(v, "pelvis_L", "pelvis_R", "upperleg01_L", "upperleg01_R", "spine05", "spine04") > 0.1:
            if not front and h > hip_h + 0.02 * torso:  # expose upper glutes
                return False
            return True
    # 2 ─ breast cups: cover lower/outer breast, plunge in the middle
    bwt = w(v, "breast_L", "breast_R")
    if bwt > 0.22:
        if abs(x) < 0.028 and h > bust_h - 0.05 * torso:   # plunge gap at sternum top
            return False
        if h > bust_h + 0.16 * torso:                       # open above (plunge neckline)
            return False
        return True
    # 3 ─ centre panel: sternum → navel → crotch, narrow
    if front and abs(x) < 0.034 and crotch_h < h < bust_h - 0.02 * torso:
        return True
    return False

# duplicate body → suit object, keep only suit faces
suit_me = me.copy(); suit_me.name = "monokini_mesh"
suit = bpy.data.objects.new("monokini", suit_me)
bpy.context.scene.collection.objects.link(suit)
suit.parent = body.parent
suit.matrix_world = body.matrix_world
am = suit.modifiers.new("Armature", "ARMATURE"); am.object = bpy.data.objects["weaver_base2"]

bm = bmesh.new(); bm.from_mesh(suit_me); bm.verts.ensure_lookup_table()
keep = set()
for f in bm.faces:
    if all(suit_region(suit_me.vertices[v.index]) for v in f.verts):
        keep.add(f.index)
bm.faces.ensure_lookup_table()
bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep], context="FACES")
# drop loose verts
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
bm.to_mesh(suit_me); bm.free()
print("suit faces:", len(suit_me.polygons))

# push suit slightly off the skin + solidify
for v in suit_me.vertices:
    v.co += v.normal * 0.0025
sol = suit.modifiers.new("Solid", "SOLIDIFY"); sol.thickness = 0.0035

# black fabric material
fab = bpy.data.materials.new("SuitBlack"); fab.use_nodes = True
b = fab.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.015, 0.015, 0.018, 1)
b.inputs["Roughness"].default_value = 0.55
try: b.inputs["Sheen Weight"].default_value = 0.4
except KeyError: pass
suit_me.materials.clear(); suit_me.materials.append(fab)

# ── gold chains ──
gold = bpy.data.materials.new("Gold"); gold.use_nodes = True
g = gold.node_tree.nodes["Principled BSDF"]
g.inputs["Base Color"].default_value = (1.0, 0.72, 0.25, 1)
g.inputs["Metallic"].default_value = 1.0
g.inputs["Roughness"].default_value = 0.18

def surface_point(h, ang):
    """Approx body surface point at height h, polar angle ang (0=front) — max radius among nearby verts."""
    best, bestr = None, 0.0
    ca, sa = math.cos(ang), math.sin(ang)
    for v in me.vertices:
        if abs(v.co[UP] - h) > 0.02: continue
        dx, dz = v.co[LAT], (v.co[DEPTH] - body_d) * FRONT
        r = math.hypot(dx, dz)
        if r < 1e-4: continue
        va = math.atan2(dx, dz)
        dd = (va - ang + math.pi) % (2 * math.pi) - math.pi
        if abs(dd) < 0.25 and r > bestr:
            bestr, best = r, Vector((math.sin(ang) * r, h, body_d + math.cos(ang) * r * FRONT))
    return best

def make_chain(name, pts, link_r=0.008):
    cur = bpy.data.curves.new(name, 'CURVE'); cur.dimensions = '3D'
    sp = cur.splines.new('NURBS'); sp.points.add(len(pts) - 1)
    for i, p in enumerate(pts): sp.points[i].co = (*p, 1)
    sp.use_endpoint_u = True
    ob = bpy.data.objects.new(name, cur)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = body.parent; ob.matrix_world = body.matrix_world
    # link unit: two perpendicular torus links
    bmt = bmesh.new()
    bmesh.ops.create_torus = None  # (no torus op) — build via spin? use primitive:
    bmt.free()
    mesh = bpy.data.meshes.new(name + "_link")
    import bpy_extras
    # simple approach: primitive_torus_add into temp object
    bpy.ops.mesh.primitive_torus_add(major_radius=link_r, minor_radius=link_r * 0.28)
    t1 = bpy.context.active_object
    bpy.ops.mesh.primitive_torus_add(major_radius=link_r, minor_radius=link_r * 0.28,
                                     rotation=(math.pi / 2, 0, 0), location=(link_r * 1.4, 0, 0))
    t2 = bpy.context.active_object
    for t in (t2,):
        bpy.ops.object.select_all(action='DESELECT')
        t1.select_set(True); t2.select_set(True)
        bpy.context.view_layer.objects.active = t1
    bpy.ops.object.join()
    unit = bpy.context.active_object
    unit.name = name + "_unit"
    unit.data.materials.append(gold)
    arr = unit.modifiers.new("Arr", "ARRAY"); arr.fit_type = 'FIT_CURVE'; arr.curve = ob
    arr.relative_offset_displace = (1.4, 0, 0)
    crv = unit.modifiers.new("Crv", "CURVE"); crv.object = ob; crv.deform_axis = 'POS_X'
    unit.parent = body.parent; unit.matrix_world = ob.matrix_world
    return ob

# belly chain: loop around hips at hip_h + 0.06*torso, slight front droop
pts = []
hch = hip_h + 0.055 * torso
for i in range(25):
    ang = -math.pi + (2 * math.pi) * i / 24
    droop = -0.035 * torso * max(0.0, math.cos(ang)) ** 2   # droops at front
    p = surface_point(hch + droop, ang)
    if p is not None:
        pts.append(p + Vector((0, 0, 0)))
make_chain("belly_chain", pts)

# strap chain: her right shoulder (x<0), from right cup top → over shoulder → upper back
sp_pts = []
for t in [i / 10 for i in range(11)]:
    h = (bust_h + 0.10 * torso) + t * (0.32 * torso)        # up over shoulder
    ang = -0.55 - t * 1.9                                    # front-right → back
    p = surface_point(min(h, bust_h + 0.30 * torso), ang)
    if p is not None: sp_pts.append(p)
make_chain("strap_chain", sp_pts, link_r=0.0065)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("OUTFIT_DONE")

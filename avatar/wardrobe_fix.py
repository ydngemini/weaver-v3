import bpy, bmesh
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

HANDS = [n for n in gi if any(k in n for k in ("finger", "metacarpal", "wrist"))]
ARMS  = [n for n in gi if any(k in n for k in ("upperarm", "lowerarm", "shoulder01"))]

UP, LAT, DEPTH = 1, 0, 2
def mean_h(names, wmin=0.15):
    hs = [v.co[UP] for v in me.vertices if w(v, *names) > wmin]
    return sum(hs) / max(len(hs), 1)
bust_h = mean_h(["breast_L", "breast_R"])
hip_h  = mean_h(["pelvis_L", "pelvis_R"])
torso  = abs(bust_h - hip_h)
body_d = sum(v.co[DEPTH] for v in me.vertices) / len(me.vertices)

# remove the three flawed tops
for nm in ("outfit_dress", "outfit_crop_top", "outfit_sport_bra"):
    ob = bpy.data.objects.get(nm)
    if ob:
        m = ob.data
        bpy.data.objects.remove(ob, do_unlink=True)
        bpy.data.meshes.remove(m)

def make_garment(name, region_fn, mat, offset=0.0045, thick=0.003, relax=4, inflate_fn=None):
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
    # drop tiny islands (speck faces)
    bm.verts.ensure_lookup_table()
    seen, isl = set(), []
    for v in bm.verts:
        if v.index in seen: continue
        stack, cur = [v], set()
        while stack:
            x = stack.pop()
            if x.index in cur: continue
            cur.add(x.index)
            for e in x.link_edges:
                o = e.other_vert(x)
                if o.index not in cur: stack.append(o)
        seen |= cur; isl.append(cur)
    big = max(len(i) for i in isl)
    kill = set().union(*[i for i in isl if len(i) < max(20, big * 0.05)]) if len(isl) > 1 else set()
    if kill:
        bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index in kill], context="VERTS")
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
        if inflate_fn: v.co += inflate_fn(v)
    sub = ob.modifiers.new("Sub", "SUBSURF"); sub.levels = 1; sub.render_levels = 2
    sol = ob.modifiers.new("Solid", "SOLIDIFY"); sol.thickness = thick
    gme.materials.clear(); gme.materials.append(mat)
    print(f"{name}: {len(gme.polygons)} faces")
    return ob

def torso_ok(v, arm_t=0.5):
    if w(v, *HANDS) > 0.10: return False
    if w(v, *ARMS) > arm_t: return False
    if w(v, "head", "neck01", "neck02", "neck03") > 0.25: return False
    return True

satin = bpy.data.materials["DressGold"]
hem_h = hip_h - 0.42 * torso
def dress_region(v):
    if not torso_ok(v): return False
    if w(v, "lowerleg01_L", "lowerleg01_R", "foot_L", "foot_R") > 0.15: return False
    # must actually be torso/upper-leg anatomy
    if w(v, "spine01","spine02","spine03","spine04","spine05","pelvis_L","pelvis_R",
         "breast_L","breast_R","upperleg01_L","upperleg01_R","upperleg02_L","upperleg02_R","root") < 0.12:
        return False
    return hem_h < v.co[UP] < bust_h + 0.10 * torso
def dress_inflate(v):
    h = v.co[UP]
    if h < hip_h:
        t = min(1.0, (hip_h - h) / (hip_h - hem_h + 1e-6))
        r = Vector((v.co[LAT], 0, v.co[DEPTH] - body_d))
        if r.length > 1e-5:
            return Vector((r.x, 0, r.z)).normalized() * (0.012 * t)
    return Vector()
make_garment("outfit_dress", dress_region, satin, inflate_fn=dress_inflate)

cotton = bpy.data.materials["TopWhite"]
def crop_region(v):
    if not torso_ok(v, 0.45): return False
    if w(v, "spine01","spine02","spine03","breast_L","breast_R","clavicle_L","clavicle_R") < 0.12: return False
    return bust_h - 0.28 * torso < v.co[UP] < bust_h + 0.24 * torso
make_garment("outfit_crop_top", crop_region, cotton)

lycra_p = bpy.data.materials["BraPurple"]
def bra_region(v):
    if not torso_ok(v, 0.45): return False
    if w(v, "spine01","spine02","spine03","breast_L","breast_R") < 0.12: return False
    return bust_h - 0.16 * torso < v.co[UP] < bust_h + 0.20 * torso
make_garment("outfit_sport_bra", bra_region, lycra_p)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("WARDROBE_FIX_DONE")

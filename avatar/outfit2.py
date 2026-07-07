"""Monokini v2 + explicit-link gold chains. Removes v1 objects first."""
import bpy, bmesh, math
from mathutils import Vector, Matrix

# ── remove v1 ──
for nm in list(bpy.data.objects.keys()):
    if any(k in nm for k in ("monokini", "belly_chain", "strap_chain", "_unit", "Torus")):
        bpy.data.objects.remove(bpy.data.objects[nm], do_unlink=True)

body = bpy.data.objects["weaver_base2-base"]
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
torso  = abs(bust_h - hip_h)
crotch_h = hip_h - 0.22 * torso
body_d = sum(v.co[DEPTH] for v in me.vertices) / len(me.vertices)

def suit_region(v):
    co = v.co; h = co[UP]; x = co[LAT]
    front = (co[DEPTH] - body_d) > 0
    if w(v, "upperarm01_L", "upperarm01_R", "lowerarm01_L", "lowerarm01_R", "wrist_L",
         "wrist_R", "head", "neck01", "neck02", "neck03", "lowerleg01_L", "lowerleg01_R") > 0.25:
        return False
    if w(v, "upperleg02_L", "upperleg02_R") > 0.45:
        return False                                        # no shorts
    # brief
    if crotch_h - 0.02 * torso < h < hip_h + 0.12 * torso:
        cut = max(0.0, (h - crotch_h) / (0.34 * torso))
        if front:
            max_x = 0.22 * (1.0 - 0.80 * min(cut, 1.0))     # narrow V front
            if abs(x) < max_x: return True
        else:
            if h < hip_h + 0.01 * torso:                    # lower glutes covered
                max_x = 0.26 * (1.0 - 0.45 * min(cut, 1.0))
                if abs(x) < max_x: return True
        return False
    # cups
    bwt = w(v, "breast_L", "breast_R")
    if bwt > 0.18:
        if h > bust_h + 0.15 * torso: return False          # plunge neckline opens up top
        if abs(x) < 0.024 and h > bust_h + 0.03 * torso: return False   # small sternum notch
        return True
    # centre panel
    if front and abs(x) < 0.034 and crotch_h - 0.02 * torso < h < bust_h + 0.02 * torso:
        return True
    return False

suit_me = me.copy(); suit_me.name = "monokini_mesh"
suit = bpy.data.objects.new("monokini", suit_me)
bpy.context.scene.collection.objects.link(suit)
suit.parent = body.parent; suit.matrix_world = body.matrix_world
am = suit.modifiers.new("Armature", "ARMATURE"); am.object = bpy.data.objects["weaver_base2"]

bm = bmesh.new(); bm.from_mesh(suit_me)
bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
keep = [f for f in bm.faces if all(suit_region(suit_me.vertices[v.index]) for v in f.verts)]
keep_set = set(f.index for f in keep)
bmesh.ops.delete(bm, geom=[f for f in bm.faces if f.index not in keep_set], context="FACES")
bmesh.ops.delete(bm, geom=[v for v in bm.verts if not v.link_faces], context="VERTS")
# relax boundary verts to kill the staircase
bm.verts.ensure_lookup_table()
boundary = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
for _ in range(3):
    upd = {}
    for v in boundary:
        nbr = [e.other_vert(v) for e in v.link_edges]
        if nbr: upd[v] = sum((n.co for n in nbr), Vector()) / len(nbr)
    for v, c in upd.items(): v.co = v.co * 0.5 + c * 0.5
bm.to_mesh(suit_me); bm.free()
print("suit faces:", len(suit_me.polygons))

for v in suit_me.vertices: v.co += v.normal * 0.003
sub = suit.modifiers.new("Sub", "SUBSURF"); sub.levels = 2; sub.render_levels = 2
sol = suit.modifiers.new("Solid", "SOLIDIFY"); sol.thickness = 0.003

fab = bpy.data.materials.new("SuitBlack2"); fab.use_nodes = True
b = fab.node_tree.nodes["Principled BSDF"]
b.inputs["Base Color"].default_value = (0.01, 0.01, 0.012, 1)
b.inputs["Roughness"].default_value = 0.5
suit_me.materials.clear(); suit_me.materials.append(fab)

# ── chains: explicit torus links along sampled surface paths ──
gold = bpy.data.materials.new("Gold2"); gold.use_nodes = True
g = gold.node_tree.nodes["Principled BSDF"]
g.inputs["Base Color"].default_value = (1.0, 0.72, 0.25, 1)
g.inputs["Metallic"].default_value = 1.0; g.inputs["Roughness"].default_value = 0.18

verts_by_band = {}
for v in me.vertices:
    verts_by_band.setdefault(round(v.co[UP], 2), []).append(v.index)

def surface_point(h, ang, out=0.006):
    """local-space surface point at height h, angle ang around vertical (0=front +z)."""
    best_r, best = 0.0, None
    for hh in (round(h,2)-0.01, round(h,2), round(h,2)+0.01):
        for vi in verts_by_band.get(hh, []):
            co = me.vertices[vi].co
            dx, dz = co[LAT], co[DEPTH] - body_d
            va = math.atan2(dx, dz)
            dd = (va - ang + math.pi) % (2*math.pi) - math.pi
            if abs(dd) < 0.22:
                r = math.hypot(dx, dz)
                if r > best_r: best_r = r
    if best_r == 0.0: return None
    r = best_r + out
    return Vector((math.sin(ang)*r, h, body_d + math.cos(ang)*r))

def build_chain(name, pts, link_r=0.0075):
    """Chain of alternating-orientation torus links along polyline pts (local space)."""
    bmc = bmesh.new()
    total = Matrix.Identity(4)
    step = link_r * 1.55
    # resample polyline at ~step spacing
    samples = []
    for i in range(len(pts)-1):
        a, b2 = pts[i], pts[i+1]
        seg = (b2-a).length
        n = max(1, int(seg/step))
        for t in range(n):
            samples.append(a + (b2-a)*(t/n))
    samples.append(pts[-1])
    for i, p in enumerate(samples):
        tang = (samples[min(i+1, len(samples)-1)] - samples[max(i-1,0)]).normalized()
        up = Vector((0,1,0))
        side = tang.cross(up).normalized() if abs(tang.dot(up)) < 0.95 else tang.cross(Vector((1,0,0))).normalized()
        rot = Matrix((tang, side, tang.cross(side))).transposed().to_4x4()
        twist = Matrix.Rotation(math.pi/2 * (i%2), 4, 'X')
        mat = Matrix.Translation(p) @ rot @ twist
        ring = bmesh.ops.create_circle(bmc, cap_ends=False, radius=link_r, segments=10)
        ringverts = ring['verts']
        # extrude circle into a thin torus-ish band: inset two circles offset along normal
        res = bmesh.ops.extrude_edge_only(bmc, edges=list({e for v in ringverts for e in v.link_edges}))
        moved = [el for el in res['geom'] if isinstance(el, bmesh.types.BMVert)]
        for v2 in moved: v2.co.z += link_r*0.35
        allv = ringverts + moved
        bmesh.ops.transform(bmc, matrix=mat, verts=allv)
    mesh = bpy.data.meshes.new(name)
    bmc.to_mesh(mesh); bmc.free()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.parent = body.parent; ob.matrix_world = body.matrix_world
    sol2 = ob.modifiers.new("Skin", "SOLIDIFY"); sol2.thickness = link_r*0.5; sol2.offset = 0
    mesh.materials.append(gold)
    return ob

# belly chain around hips
pts = []
hch = hip_h + 0.05 * torso
for i in range(37):
    ang = -math.pi + (2*math.pi)*i/36
    droop = -0.05 * torso * max(0.0, math.cos(ang))**3
    p = surface_point(hch + droop, ang)
    if p: pts.append(p)
if len(pts) > 4: build_chain("belly_chain", pts + [pts[0]])
print("belly pts:", len(pts))

# strap chain over her right shoulder (x<0): cup top front → shoulder → back
pts2 = []
sh_h = bust_h + 0.30 * torso     # shoulder height approx
for i in range(13):
    t = i/12
    # angle sweeps from front-right (-0.5 rad) around over the shoulder to back-right (-2.6)
    ang = -0.5 - 2.1*t
    h = bust_h + 0.10*torso + (sh_h - bust_h - 0.10*torso) * math.sin(math.pi*t)
    p = surface_point(h, ang, out=0.005)
    if p: pts2.append(p)
if len(pts2) > 4: build_chain("strap_chain", pts2, link_r=0.006)
print("strap pts:", len(pts2))

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("OUTFIT2_DONE")

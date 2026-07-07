"""Weaver avatar reshape pass 1 — bust/waist/glutes/hips/thighs + african skin + previews.

Runs headless:  blender --background work.blend --python reshape.py
Deformation is weight-driven (MH rig weights as soft selections) in LOCAL mesh space;
all landmarks (bust/hip heights, front/back sign) are derived from the mesh itself.
Tunables in K below — iterate K, re-run, re-render.
"""
import bpy, math
from mathutils import Vector

K = dict(
    BUST_PUSH=0.030,      # normal-push on breast-weighted verts (32G target)
    BUST_DROP=0.005, BUST_CLEAVAGE=0.014,      # slight gravity drop
    WAIST_SHRINK=0.20,    # 20% radial slim at band center
    BELLY_FLATTEN=0.012,  # extra inward push on front belly verts
    GLUTE_PUSH=0.050,     # normal-push on posterior pelvis/upperleg verts
    GLUTE_BACK=0.014,     # extra straight-backward bias
    HIP_WIDEN=0.045,       # lateral scale gain in hip band
    THIGH_PUSH=0.016,     # thigh fill, fades down the leg
)

SKIN_DIFFUSE = "/usr/share/makehuman-community/data/skins/textures/young_darkskinned_female_diffuse.png"
OUTDIR = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders"

body = bpy.data.objects["weaver_base2-base"]
me = body.data

# ── clean leftover test object ──
if "Sphere" in bpy.data.objects:
    bpy.data.objects.remove(bpy.data.objects["Sphere"], do_unlink=True)

# ── weight lookup helpers ──
gi = {g.name: g.index for g in body.vertex_groups}
def w(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]:
                s += g.weight
    return min(s, 1.0)

ARM = [n for n in gi if any(k in n for k in ("shoulder01", "upperarm", "lowerarm",
       "wrist", "finger", "metacarpal", "clavicle"))]
LEG_LOW = [n for n in gi if any(k in n for k in ("lowerleg", "foot", "toe"))]

# ── local axes: height = max bbox extent, depth = min ──
dims = body.dimensions  # proportional to local bbox * scale
ext = [(me.vertices[0].co[i], me.vertices[0].co[i]) for i in range(3)]
for v in me.vertices:
    for i in range(3):
        ext[i] = (min(ext[i][0], v.co[i]), max(ext[i][1], v.co[i]))
spans = [e[1] - e[0] for e in ext]
UP = spans.index(max(spans))          # height axis
DEPTH = spans.index(min(spans))       # front-back axis
LAT = ({0, 1, 2} - {UP, DEPTH}).pop() # lateral axis

# ── landmarks from weights ──
def mean_h(names, wmin=0.15):
    hs = [v.co[UP] for v in me.vertices if w(v, *names) > wmin]
    return sum(hs) / max(len(hs), 1)
def mean_d(names, wmin=0.15):
    ds = [v.co[DEPTH] for v in me.vertices if w(v, *names) > wmin]
    return sum(ds) / max(len(ds), 1)

bust_h = mean_h(["breast_L", "breast_R"])
hip_h = mean_h(["pelvis_L", "pelvis_R"])
body_d = sum(v.co[DEPTH] for v in me.vertices) / len(me.vertices)
front_sign = 1.0 if mean_d(["breast_L", "breast_R"]) > body_d else -1.0

torso = abs(bust_h - hip_h)
waist_h = hip_h + 0.58 * torso
waist_band = 0.34 * torso
hip_band = 0.42 * torso

normals = [v.normal.copy() for v in me.vertices]

def band(h, center, half):        # smooth cosine falloff 1→0
    t = abs(h - center) / half
    return 0.0 if t >= 1.0 else 0.5 + 0.5 * math.cos(math.pi * t)

for v in me.vertices:
    co, n = v.co, normals[v.index]
    arm_w = w(v, *ARM)
    if arm_w > 0.6:
        continue                                   # never deform arms/hands
    torso_gate = 1.0 - arm_w

    # 1 ─ bust
    bw = w(v, "breast_L", "breast_R")
    if bw > 0.0:
        bs = bw * bw * (3 - 2 * bw)              # smoothstep: feather the shelf
        co += n * (K["BUST_PUSH"] * bs)
        co[UP] -= K["BUST_DROP"] * bs
        co[LAT] *= 1.0 - K["BUST_CLEAVAGE"] * bs # pull toward centre = cleavage

    # 2 ─ waist slim + belly flatten
    fb = band(co[UP], waist_h, waist_band) * torso_gate
    if fb > 0.0 and w(v, *LEG_LOW) < 0.2:
        shrink = 1.0 - K["WAIST_SHRINK"] * fb
        co[LAT] *= shrink                          # body is lateral-centred at 0
        co[DEPTH] = body_d + (co[DEPTH] - body_d) * shrink
        if (co[DEPTH] - body_d) * front_sign > 0:  # front-of-body: flatten belly
            co[DEPTH] -= front_sign * K["BELLY_FLATTEN"] * fb

    # 3 ─ glutes (posterior pelvis/upper-thigh verts)
    gw = w(v, "pelvis_L", "pelvis_R", "upperleg01_L", "upperleg01_R")
    post = (body_d - co[DEPTH]) * front_sign     # >0 means behind body centre
    if gw > 0.05 and post > 0.015:
        g = gw * band(co[UP], hip_h, hip_band)
        lat_damp = max(0.0, 1.0 - abs(co[LAT]) / (0.55 * hip_band + 1e-6)) ** 0.5
        g *= 0.35 + 0.65 * lat_damp              # keep it on the cheeks, off the sides
        co += n * (K["GLUTE_PUSH"] * g)
        co[DEPTH] -= front_sign * K["GLUTE_BACK"] * g

    # 4 ─ hip widen
    hb = band(co[UP], hip_h, hip_band) * torso_gate
    if hb > 0.0:
        co[LAT] *= 1.0 + K["HIP_WIDEN"] * hb

    # 5 ─ thighs
    tw = w(v, "upperleg01_L", "upperleg01_R", "upperleg02_L", "upperleg02_R")
    if tw > 0.05:
        co += n * (K["THIGH_PUSH"] * tw)

me.update()

# ── crease relaxation on deformed regions ──
sm = body.vertex_groups.new(name="deform_mask")
idxs = [v.index for v in me.vertices
        if w(v, "breast_L", "breast_R", "pelvis_L", "pelvis_R",
             "upperleg01_L", "upperleg01_R", "spine03", "spine04") > 0.05]
sm.add(idxs, 1.0, "REPLACE")
mod = body.modifiers.new("Relax", "SMOOTH")
mod.factor = 0.6; mod.iterations = 4; mod.vertex_group = "deform_mask"
bpy.context.view_layer.objects.active = body
bpy.ops.object.modifier_apply(modifier="Relax")

# ── relink eye texture (was rendering magenta) ──
EYE_TEX = "/home/ydn/Documents/makehuman/v1py3/exports/textures/brown_eye.png"
import os as _os
for img in bpy.data.images:
    if "brown_eye" in img.name.lower() and _os.path.exists(EYE_TEX):
        img.filepath = EYE_TEX
        img.reload()

# ── skin material ──
mat = bpy.data.materials.get("DefaultSkin")
mat.use_nodes = True
nt = mat.node_tree
nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.load(SKIN_DIFFUSE)
bsdf.inputs["Roughness"].default_value = 0.45
mix = nt.nodes.new("ShaderNodeMix"); mix.data_type = 'RGBA'
mix.blend_type = 'MULTIPLY'; mix.inputs["Factor"].default_value = 0.55
mix.inputs[7].default_value = (0.42, 0.33, 0.30, 1.0)  # deep-tone multiply
nt.links.new(tex.outputs["Color"], mix.inputs[6])
nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])

# ── previews: front / three-quarter / back ──
import os
os.makedirs(OUTDIR, exist_ok=True)
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = sc.render.resolution_y = 1024
sc.world = sc.world or bpy.data.worlds.new("W")
sc.world.use_nodes = True
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.12, 0.12, 0.13, 1)

# world-space bbox
mw = body.matrix_world
corners = [mw @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center = (cmin + cmax) / 2
wspans = cmax - cmin
WUP = max(range(3), key=lambda i: wspans[i])
height = wspans[WUP]
horiz = [i for i in range(3) if i != WUP]

cam = bpy.data.objects.new("PrevCam", bpy.data.cameras.new("PrevCam"))
sc.collection.objects.link(cam)
sc.camera = cam
key = bpy.data.objects.new("Key", bpy.data.lights.new("Key", "SUN")); key.data.energy = 3.5
fill = bpy.data.objects.new("Fill", bpy.data.lights.new("Fill", "SUN")); fill.data.energy = 1.2
sc.collection.objects.link(key); sc.collection.objects.link(fill)

def look_at(obj, target):
    d = target - obj.location
    obj.rotation_euler = d.to_track_quat("-Z", "Y").to_euler()

dist = height * 1.35
wfront = mw.to_3x3() @ Vector([front_sign if i == DEPTH else 0 for i in range(3)])
wfront.normalize()
wup = Vector([1 if i == WUP else 0 for i in range(3)])
wside = wfront.cross(wup)

for name, yaw in (("front", 0), ("threequarter", 40), ("back", 180)):
    a = math.radians(yaw)
    d = (wfront * math.cos(a) + wside * math.sin(a)).normalized()
    cam.location = center + d * dist + wup * height * 0.02
    look_at(cam, center)
    key.location = center + (d + wside * 0.8 + wup * 0.9).normalized() * dist
    look_at(key, center)
    fill.location = center + (d - wside * 0.9 + wup * 0.2).normalized() * dist
    look_at(fill, center)
    sc.render.filepath = f"{OUTDIR}/pass2_{name}.png"
    bpy.ops.render.render(write_still=True)
    print("RENDERED", sc.render.filepath)

# head close-up (front)
cam.location = center + wfront * (height * 0.55) + wup * height * 0.335
head_target = center + wup * height * 0.36
look_at(cam, head_target)
sc.render.filepath = f"{OUTDIR}/pass2_head.png"
bpy.ops.render.render(write_still=True)
print("RENDERED", sc.render.filepath)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("RESHAPE_DONE bust_h=%.3f hip_h=%.3f waist_h=%.3f UP=%d LAT=%d DEPTH=%d front=%+.0f"
      % (bust_h, hip_h, waist_h, UP, LAT, DEPTH, front_sign))

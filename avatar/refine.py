import bpy, math
from mathutils import Vector

OUT = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders"
body = bpy.data.objects["weaver_base2-base"]
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
arm  = bpy.data.objects["weaver_base2"]
me = body.data

# ── 1. seat the eyeballs into the sockets ──
mw = eyes.matrix_world
bb = [mw @ Vector(c) for c in eyes.bound_box]
cur = sum(bb, Vector()) / 8
eyeL = arm.matrix_world @ arm.data.bones["eye_L"].head_local
eyeR = arm.matrix_world @ arm.data.bones["eye_R"].head_local
target = (eyeL + eyeR) / 2
delta_world = target - cur
delta_local = mw.to_3x3().inverted() @ delta_world
for v in eyes.data.vertices:
    v.co += delta_local
eyes.data.update()
print("EYES moved by world", [round(x,3) for x in delta_world])

# ── 2. light face feminization ──
gi = {g.name: g.index for g in body.vertex_groups}
def w(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]:
                s += g.weight
    return min(s, 1.0)

# local axes as before: UP=1(y), LAT=0(x), DEPTH=2(z) in mesh local space
UP, LAT, DEPTH = 1, 0, 2
head_c = None
hs = [v.co for v in me.vertices if w(v, "head") > 0.5]
if hs:
    head_c = sum(hs, Vector()) / len(hs)
normals = [v.normal.copy() for v in me.vertices]
for v in me.vertices:
    co, n = v.co, normals[v.index]
    jw = w(v, "jaw")
    if jw > 0.05 and head_c is not None:
        # taper jaw toward centreline; soften chin forward point
        co[LAT] = head_c[LAT] + (co[LAT] - head_c[LAT]) * (1.0 - 0.10 * jw)
    lw = w(v, "oris02", "oris03_L", "oris03_R", "oris04_L", "oris04_R",
           "oris05", "oris06", "oris06_L", "oris06_R", "oris07_L", "oris07_R")
    if lw > 0.05:
        co += n * (0.004 * lw)          # fuller lips
    bw = w(v, "oculi01_L", "oculi01_R") # brow ridge soften (pull slightly back)
    if bw > 0.1:
        co += n * (-0.0025 * bw)
me.update()
print("FACE refined")

# ── 3. head + full-body re-render with fixed lighting ──
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = sc.render.resolution_y = 1024
cam = bpy.data.objects["PrevCam"]
key = bpy.data.objects["Key"]; fill = bpy.data.objects["Fill"]

mwb = body.matrix_world
corners = [mwb @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center = (cmin + cmax) / 2
height = (cmax - cmin).z
wfront = Vector((0, -1, 0))   # verified: front = -Y world (breasts at y<0)
wup = Vector((0, 0, 1))
wside = wfront.cross(wup)

def look_at(obj, t):
    obj.rotation_euler = (t - obj.location).to_track_quat("-Z", "Y").to_euler()

def light_rig(view_dir, focus):
    key.location = focus + (view_dir + wside * 0.7 + wup * 0.8).normalized() * height
    look_at(key, focus)
    fill.location = focus + (view_dir - wside * 0.8 + wup * 0.3).normalized() * height
    look_at(fill, focus)

# head closeup
head_t = center + wup * height * 0.405
cam.location = head_t + wfront * (height * 0.28)
look_at(cam, head_t)
light_rig(wfront, head_t)
sc.render.filepath = f"{OUT}/pass3_head.png"
bpy.ops.render.render(write_still=True); print("RENDERED", sc.render.filepath)

# full front
cam.location = center + wfront * (height * 1.35)
look_at(cam, center)
light_rig(wfront, center)
sc.render.filepath = f"{OUT}/pass3_front.png"
bpy.ops.render.render(write_still=True); print("RENDERED", sc.render.filepath)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("REFINE_DONE")

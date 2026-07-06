import bpy, math
from mathutils import Vector
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = 1440; sc.render.resolution_y = 1080
sc.frame_set(1)
# monokini outfit visible
SHOW = ["monokini", "belly_chain", "strap_chain"]
for nm in ["outfit_dress","outfit_crop_top","outfit_jeans","outfit_sport_bra","outfit_leggings"] + SHOW:
    ob = bpy.data.objects.get(nm)
    if ob: ob.hide_render = nm not in SHOW
# kill the old studio suns (apartment has its own lighting)
for nm in ("Key", "Fill"):
    ob = bpy.data.objects.get(nm)
    if ob: ob.hide_render = True
cam = bpy.data.objects["PrevCam"]
cam.data.lens = 32
def look_at(o, t): o.rotation_euler = (t - o.location).to_track_quat("-Z", "Y").to_euler()

# shot 1: wide cinematic — her + room + city window
cam.location = Vector((3.4, 2.7, 1.75))
look_at(cam, Vector((-0.4, -1.2, 1.05)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_wide.png"
bpy.ops.render.render(write_still=True); print("RENDERED wide")

# shot 2: her against the window, city behind
cam.location = Vector((-1.1, 1.9, 1.45))
look_at(cam, Vector((0.05, -1.4, 1.0)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_hero.png"
bpy.ops.render.render(write_still=True); print("RENDERED hero")

# shot 3: detail — figurine pedestal + window corner
cam.location = Vector((2.2, -0.9, 1.35)); cam.data.lens = 40
look_at(cam, Vector((3.9, -2.6, 1.15)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_figurine.png"
bpy.ops.render.render(write_still=True); print("RENDERED figurine")

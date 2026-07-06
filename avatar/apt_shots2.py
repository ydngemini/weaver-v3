import bpy
from mathutils import Vector
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = 1440; sc.render.resolution_y = 1080
sc.frame_set(1)
SHOW = ["monokini", "belly_chain", "strap_chain"]
for nm in ["outfit_dress","outfit_crop_top","outfit_jeans","outfit_sport_bra","outfit_leggings"] + SHOW:
    ob = bpy.data.objects.get(nm)
    if ob: ob.hide_render = nm not in SHOW
for nm in ("Key", "Fill"):
    ob = bpy.data.objects.get(nm)
    if ob: ob.hide_render = True
cam = bpy.data.objects["PrevCam"]
def look_at(o, t): o.rotation_euler = (t - o.location).to_track_quat("-Z", "Y").to_euler()

# hero: from window side looking in — her front-lit, room behind
cam.data.lens = 35
cam.location = Vector((0.9, -3.1, 1.35))
look_at(cam, Vector((0.0, -0.6, 1.0)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_hero.png"
bpy.ops.render.render(write_still=True); print("RENDERED hero")

# wide: cinematic room + her + window
cam.data.lens = 28
cam.location = Vector((3.6, 2.6, 1.9))
look_at(cam, Vector((-0.6, -1.5, 1.0)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_wide.png"
bpy.ops.render.render(write_still=True); print("RENDERED wide")

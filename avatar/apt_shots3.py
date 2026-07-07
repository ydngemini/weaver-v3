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

cam.data.lens = 24
cam.location = Vector((3.9, -2.9, 2.0))            # from window corner: kitchen+living+her
look_at(cam, Vector((-1.2, 1.6, 0.9)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_full.png"
bpy.ops.render.render(write_still=True); print("RENDERED full")

cam.data.lens = 30
cam.location = Vector((-1.2, -2.6, 1.6))           # bedroom zone
look_at(cam, Vector((-3.6, 1.5, 0.55)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_bedroom.png"
bpy.ops.render.render(write_still=True); print("RENDERED bedroom")

cam.location = Vector((0.4, -1.0, 1.55))           # kitchen + bar
look_at(cam, Vector((3.6, 1.5, 0.9)))
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/apt_kitchen.png"
bpy.ops.render.render(write_still=True); print("RENDERED kitchen")

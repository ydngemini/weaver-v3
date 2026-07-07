import bpy, math
from mathutils import Vector
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"; sc.render.resolution_x = sc.render.resolution_y = 1024
# show monokini outfit
SHOW = ["monokini", "belly_chain", "strap_chain"]
for nm in ["outfit_dress","outfit_crop_top","outfit_jeans","outfit_sport_bra","outfit_leggings"] + SHOW:
    ob = bpy.data.objects.get(nm)
    if ob: ob.hide_render = nm not in SHOW
body = bpy.data.objects["weaver_base2-base"]
cam = bpy.data.objects["PrevCam"]; key = bpy.data.objects["Key"]; fill = bpy.data.objects["Fill"]
mwb = body.matrix_world
corners = [mwb @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center = (cmin+cmax)/2; height = (cmax-cmin).z
wfront = Vector((0,-1,0)); wup = Vector((0,0,1)); wside = wfront.cross(wup)
def look_at(o,t): o.rotation_euler=(t-o.location).to_track_quat("-Z","Y").to_euler()
a = math.radians(20)
d = (wfront*math.cos(a)+wside*math.sin(a)).normalized()
cam.location = center + d*height*1.32; look_at(cam, center)
key.location = center + (d+wside*0.7+wup*0.8).normalized()*height; look_at(key, center)
fill.location = center + (d-wside*0.8+wup*0.3).normalized()*height; look_at(fill, center)
for f in (1, 60):
    sc.frame_set(f)
    sc.render.filepath = f"/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/idle_f{f:03d}.png"
    bpy.ops.render.render(write_still=True); print("RENDERED", f)

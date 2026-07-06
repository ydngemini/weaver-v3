import bpy, math, sys
from mathutils import Vector
prefix = [a for a in sys.argv[sys.argv.index("--")+1:]][0] if "--" in sys.argv else "view"
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"; sc.render.resolution_x = sc.render.resolution_y = 1024
body = bpy.data.objects["weaver_base2-base"]
cam = bpy.data.objects["PrevCam"]; key = bpy.data.objects["Key"]; fill = bpy.data.objects["Fill"]
mwb = body.matrix_world
corners = [mwb @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center = (cmin+cmax)/2; height = (cmax-cmin).z
wfront = Vector((0,-1,0)); wup = Vector((0,0,1)); wside = wfront.cross(wup)
def look_at(o,t): o.rotation_euler=(t-o.location).to_track_quat("-Z","Y").to_euler()
for name, yaw in (("front",0),("threequarter",40),("back",180)):
    a = math.radians(yaw)
    d = (wfront*math.cos(a)+wside*math.sin(a)).normalized()
    cam.location = center + d*height*1.35
    look_at(cam, center)
    key.location = center + (d+wside*0.7+wup*0.8).normalized()*height; look_at(key, center)
    fill.location = center + (d-wside*0.8+wup*0.3).normalized()*height; look_at(fill, center)
    sc.render.filepath = f"/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/{prefix}_{name}.png"
    bpy.ops.render.render(write_still=True); print("RENDERED", name)

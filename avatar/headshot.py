import bpy
from mathutils import Vector
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = sc.render.resolution_y = 1024
body = bpy.data.objects["weaver_base2-base"]
cam = bpy.data.objects["PrevCam"]; key = bpy.data.objects["Key"]; fill = bpy.data.objects["Fill"]
mwb = body.matrix_world
corners = [mwb @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center=(cmin+cmax)/2; height=(cmax-cmin).z
wfront=Vector((0,-1,0)); wup=Vector((0,0,1)); wside=wfront.cross(wup)
def look_at(o,t): o.rotation_euler=(t-o.location).to_track_quat("-Z","Y").to_euler()
head_t = center + wup*height*0.405
cam.location = head_t + wfront*(height*0.30) + wup*height*0.01
look_at(cam, head_t)
key.location = head_t + (wfront+wside*0.7+wup*0.5).normalized()*height; look_at(key, head_t)
fill.location = head_t + (wfront-wside*0.8+wup*0.1).normalized()*height; look_at(fill, head_t)
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/pass4_head.png"
bpy.ops.render.render(write_still=True); print("RENDERED")

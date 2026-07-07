import bpy, math
from mathutils import Vector

sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"
sc.render.resolution_x = sc.render.resolution_y = 1280
body = bpy.data.objects["weaver_base2-base"]
cam = bpy.data.objects["PrevCam"]; key = bpy.data.objects["Key"]; fill = bpy.data.objects["Fill"]
mwb = body.matrix_world
corners = [mwb @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center = (cmin+cmax)/2; height = (cmax-cmin).z
wfront = Vector((0,-1,0)); wup = Vector((0,0,1)); wside = wfront.cross(wup)
def look_at(o,t): o.rotation_euler=(t-o.location).to_track_quat("-Z","Y").to_euler()
for i in range(8):
    a = math.radians(i*45)
    d = (wfront*math.cos(a)+wside*math.sin(a)).normalized()
    cam.location = center + d*height*1.3
    look_at(cam, center)
    key.location = center + (d+wside*0.7+wup*0.8).normalized()*height; look_at(key, center)
    fill.location = center + (d-wside*0.8+wup*0.3).normalized()*height; look_at(fill, center)
    sc.render.filepath = f"/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/turn_{i*45:03d}.png"
    bpy.ops.render.render(write_still=True)
print("TURNTABLE_DONE")

# GLB export: hide render-rig objects, export meshes + armature
for nm in ("PrevCam", "Key", "Fill"):
    ob = bpy.data.objects.get(nm)
    if ob: bpy.data.objects.remove(ob, do_unlink=True)
bpy.ops.export_scene.gltf(
    filepath="/media/ydn/SYPHER_CORE2/weaver v3/avatar/weaver_avatar.glb",
    export_format='GLB',
    export_apply=True,
    export_animations=False,
    export_skins=True,
    export_yup=True,
)
print("GLB_DONE")

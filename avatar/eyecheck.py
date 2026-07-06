import bpy
from mathutils import Vector
body = bpy.data.objects["weaver_base2-base"]
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
for ob in (body, eyes):
    mw = ob.matrix_world
    bb = [mw @ Vector(c) for c in ob.bound_box]
    mn = Vector((min(c[i] for c in bb) for i in range(3)))
    mx = Vector((max(c[i] for c in bb) for i in range(3)))
    print(ob.name, "| parent:", ob.parent.name if ob.parent else None,
          "| loc:", [round(x,3) for x in ob.location],
          "| world bbox min:", [round(x,3) for x in mn], "max:", [round(x,3) for x in mx])
# head bone world pos
arm = bpy.data.objects["weaver_base2"]
hb = arm.data.bones.get("head")
if hb: print("head bone world:", [round(x,3) for x in (arm.matrix_world @ hb.head_local)])
eb_l = arm.data.bones.get("eye_L"); eb_r = arm.data.bones.get("eye_R")
for nm,b in (("eye_L",eb_l),("eye_R",eb_r)):
    if b: print(nm, "bone world:", [round(x,3) for x in (arm.matrix_world @ b.head_local)])

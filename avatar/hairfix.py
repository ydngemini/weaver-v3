import bpy
from mathutils import Vector, Matrix

body = bpy.data.objects["weaver_base2-base"]
hair = bpy.data.objects["hair_braids"]
arm = bpy.data.objects["weaver_base2"]
me = body.data
gi = {g.name: g.index for g in body.vertex_groups}
def w(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]: s += g.weight
    return min(s, 1.0)

# re-parent to armature OBJECT with clean transforms
hair.parent = None
hair.matrix_parent_inverse = Matrix.Identity(4)
hair.parent = arm
hair.matrix_parent_inverse = arm.matrix_world.inverted()

# head bbox in world
bwm = body.matrix_world
hv = [bwm @ v.co for v in me.vertices if w(v, "head") > 0.5]
hmin = Vector((min(p[i] for p in hv) for i in range(3)))
hmax = Vector((max(p[i] for p in hv) for i in range(3)))
hc = (hmin + hmax) / 2

# wipe transform and rebuild: scale, then center cap on skull
s = 0.112
hair.matrix_world = Matrix.Translation(Vector((0,0,0))) @ Matrix.Scale(s, 4)
bpy.context.view_layer.update()
bb = [hair.matrix_world @ Vector(c) for c in hair.bound_box]
bmin = Vector((min(c[i] for c in bb) for i in range(3)))
bmax = Vector((max(c[i] for c in bb) for i in range(3)))
delta = Vector((hc.x - (bmin.x+bmax.x)/2,
                hc.y - (bmin.y+bmax.y)/2 - 0.01,     # slight backward bias
                (hmax.z + 0.008) - bmax.z))
hair.matrix_world = Matrix.Translation(delta) @ hair.matrix_world
bpy.context.view_layer.update()
bb = [hair.matrix_world @ Vector(c) for c in hair.bound_box]
print("hair bbox now:", [round(min(c[i] for c in bb),3) for i in range(3)],
      [round(max(c[i] for c in bb),3) for i in range(3)])
print("head bbox:    ", [round(x,3) for x in hmin], [round(x,3) for x in hmax])
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("HAIRFIX_DONE")

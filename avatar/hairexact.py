import bpy, math
from mathutils import Vector, Matrix

body = bpy.data.objects["weaver_base2-base"]
hair = bpy.data.objects["hair_braids"]
dg = bpy.context.evaluated_depsgraph_get()

def eval_pts(ob, idxs):
    ev = ob.evaluated_get(dg); m = ev.to_mesh(); mw = ev.matrix_world
    out = [mw @ m.vertices[i].co for i in idxs]
    ev.to_mesh_clear(); return out

# ── fit rigid deform D: rest-world → render-world (hair is 100% head-weighted) ──
n = len(hair.data.vertices)
samp = [0, n//3, 2*n//3, n-1]
rest = [hair.matrix_world @ hair.data.vertices[i].co for i in samp]
ev   = eval_pts(hair, samp)
R = Matrix((( (rest[1]-rest[0]) ), (rest[2]-rest[0]), (rest[3]-rest[0]))).transposed()
E = Matrix((( (ev[1]-ev[0]) ), (ev[2]-ev[0]), (ev[3]-ev[0]))).transposed()
Drot = E @ R.inverted()
Dt = ev[0] - Drot @ rest[0]
D = Matrix.Translation(Dt) @ Drot.to_4x4()

# rendered hair bbox
allv = list(range(0, n, max(1, n//400)))
hpts = eval_pts(hair, allv)
hmin = Vector((min(p[i] for p in hpts) for i in range(3)))
hmax = Vector((max(p[i] for p in hpts) for i in range(3)))
hc = (hmin + hmax) / 2

# rendered head bbox (evaluated body verts weighted to head)
gi = {g.name: g.index for g in body.vertex_groups}
def w(v):
    for g in v.groups:
        if g.group == gi["head"]: return g.weight
    return 0.0
hidx = [v.index for v in body.data.vertices if w(v) > 0.5]
bpts = eval_pts(body, hidx)
bmin = Vector((min(p[i] for p in bpts) for i in range(3)))
bmax = Vector((max(p[i] for p in bpts) for i in range(3)))
bc = (bmin + bmax) / 2
print("rendered head bbox:", [round(x,3) for x in bmin], [round(x,3) for x in bmax])
print("rendered hair bbox:", [round(x,3) for x in hmin], [round(x,3) for x in hmax])

# ── desired delta in RENDER space ──
# 1) rotate 180° about vertical through hair center (open the face)
T = Matrix.Translation(hc) @ Matrix.Rotation(math.pi, 4, 'Z') @ Matrix.Translation(-hc)
# 2) then align: hair center xy → head center xy (+0.015 back), hair top → head top + 0.01
newc = T @ hc  # unchanged by rotation about itself
d = Vector((bc.x - hc.x, bc.y - hc.y + 0.018, (bmax.z + 0.010) - hmax.z))
T = Matrix.Translation(d) @ T

# apply in rest space: M' = D⁻¹ T D M
hair.matrix_world = D.inverted() @ T @ D @ hair.matrix_world
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("HAIREXACT_DONE")

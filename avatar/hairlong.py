import bpy, math
from mathutils import Vector, Matrix

body = bpy.data.objects["weaver_base2-base"]
hair = bpy.data.objects["hair_braids"]
dg = bpy.context.evaluated_depsgraph_get()
me = hair.data
n = len(me.vertices)

def eval_pts(ob, idxs):
    ev = ob.evaluated_get(dg); m = ev.to_mesh(); mw = ev.matrix_world
    out = [mw @ m.vertices[i].co for i in idxs]
    ev.to_mesh_clear(); return out

# fit rigid deform D (rest→render) as before
samp = [0, n//3, 2*n//3, n-1]
rest = [hair.matrix_world @ me.vertices[i].co for i in samp]
ev = eval_pts(hair, samp)
R = Matrix(((rest[1]-rest[0]), (rest[2]-rest[0]), (rest[3]-rest[0]))).transposed()
E = Matrix(((ev[1]-ev[0]), (ev[2]-ev[0]), (ev[3]-ev[0]))).transposed()
D = Matrix.Translation(ev[0] - (E @ R.inverted()) @ rest[0]) @ (E @ R.inverted()).to_4x4()

# current rendered bbox
allv = list(range(0, n, max(1, n//500)))
hp = eval_pts(hair, allv)
hmin = Vector((min(p[i] for p in hp) for i in range(3)))
hmax = Vector((max(p[i] for p in hp) for i in range(3)))
hc = (hmin + hmax) / 2

# 1) slightly forward + 6% bigger (render space)
T = Matrix.Translation(Vector((0, -0.02, 0.005))) @ \
    Matrix.Translation(hc) @ Matrix.Scale(1.06, 4) @ Matrix.Translation(-hc)
hair.matrix_world = D.inverted() @ T @ D @ hair.matrix_world
bpy.context.view_layer.update()
dg.update()

# 2) elongate braids: stretch verts below the cap line (render z), k=1.85
hp = eval_pts(hair, allv)
zmax = max(p.z for p in hp)
capline_render = zmax - 0.085
# map capline into REST space via D⁻¹ (rigid): work per-vertex in rest world
Dinv = D.inverted()
cap_rest_z = (Dinv @ Vector((0, 0, capline_render))).z   # only valid if D has no tilt — check
# safer: transform two probe points to get the rest-space direction of render-z
p0 = Dinv @ Vector((0, 0, 0)); p1 = Dinv @ Vector((0, 0, 1))
axis = (p1 - p0).normalized()          # rest-space direction of render vertical
# per-vertex: rest world pos, project onto axis, stretch below cap projection
cap_proj = (Dinv @ Vector((hc.x, hc.y, capline_render))).dot(axis)
mwl = hair.matrix_world.inverted()
k = 1.85
for v in me.vertices:
    wp = hair.matrix_world @ v.co
    t = wp.dot(axis)
    if t < cap_proj:
        wp = wp + axis * ((t - cap_proj) * (k - 1.0))
        v.co = mwl @ wp
me.update()
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("HAIRLONG_DONE")

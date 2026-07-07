import bpy, bmesh, math
from mathutils import Vector, Matrix, Euler

eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
me = eyes.data
bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
seen, islands = set(), []
for v in bm.verts:
    if v.index in seen: continue
    stack, isl = [v], set()
    while stack:
        cur = stack.pop()
        if cur.index in isl: continue
        isl.add(cur.index)
        for e in cur.link_edges:
            o = e.other_vert(cur)
            if o.index not in isl: stack.append(o)
    seen |= isl; islands.append(sorted(isl))
bm.free()

# rotate each ball: pitch irises down/forward. World rotation mapped to local.
mwl = eyes.matrix_world.to_3x3().inverted()
Rw = Euler((math.radians(-24), 0, 0), 'XYZ').to_matrix()   # pitch down around world X
Rl = mwl @ Rw @ eyes.matrix_world.to_3x3()
for isl in islands:
    c = Vector()
    for i in isl: c += me.vertices[i].co
    c /= len(isl)
    for i in isl:
        me.vertices[i].co = c + Rl @ (me.vertices[i].co - c)
me.update()
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("GAZE_DONE")

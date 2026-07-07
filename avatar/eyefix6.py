import bpy, bmesh
from mathutils import Vector

body = bpy.data.objects["weaver_base2-base"]
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
me, bme = eyes.data, body.data
dg = bpy.context.evaluated_depsgraph_get()

gi = {g.name: g.index for g in body.vertex_groups}
def bw(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]: s += g.weight
    return s

lidL = [v.index for v in bme.vertices if bw(v, "orbicularis03_L", "orbicularis04_L") > 0.3]
lidR = [v.index for v in bme.vertices if bw(v, "orbicularis03_R", "orbicularis04_R") > 0.3]
print("eyelid vert counts L/R:", len(lidL), len(lidR))

def eval_center(ob, idxs):
    ev = ob.evaluated_get(dg); evme = ev.to_mesh(); mw = ev.matrix_world
    c = Vector()
    for i in idxs: c += mw @ evme.vertices[i].co
    c /= len(idxs); ev.to_mesh_clear(); return c

tL = eval_center(body, lidL); tR = eval_center(body, lidR)
print("true socket centers L", [round(x,4) for x in tL], "R", [round(x,4) for x in tR])

# island groups (same as before)
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
def rest_center(idxs):
    c = Vector()
    for i in idxs: c += me.vertices[i].co
    return c / len(idxs)
cs = [(rest_center(i), i) for i in islands]
xs = sorted(c.x for c, _ in cs); split = (xs[0] + xs[-1]) / 2
flatA = [x for c, i in cs if c.x <= split for x in i]
flatB = [x for c, i in cs if c.x > split for x in i]

cA = eval_center(eyes, flatA); cB = eval_center(eyes, flatB)
pairs = [(flatA, tL), (flatB, tR)] if (cA-tL).length+(cB-tR).length <= (cA-tR).length+(cB-tL).length else [(flatA, tR), (flatB, tL)]

# recess into head: from socket center, inward = +Y world (front=-Y), ~0.008
mw3 = eyes.matrix_world.to_3x3().inverted()
for idxs, tgt in pairs:
    tgt = tgt + Vector((0, 0.008, 0))
    for it in range(5):
        cur = eval_center(eyes, idxs); d = tgt - cur
        if d.length < 4e-4: break
        for i in idxs: me.vertices[i].co += mw3 @ d
        me.update(); dg.update()
    print("seated:", [round(x,4) for x in eval_center(eyes, idxs)], "→", [round(x,4) for x in tgt])

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX6_DONE")

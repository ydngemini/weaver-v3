import bpy, bmesh
from mathutils import Vector

body = bpy.data.objects["weaver_base2-base"]
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
arm  = bpy.data.objects["weaver_base2"]
me, bme = eyes.data, body.data
dg = bpy.context.evaluated_depsgraph_get()

# mesh islands via bmesh
bm = bmesh.new(); bm.from_mesh(me)
bm.verts.ensure_lookup_table()
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
print("islands:", [(len(i)) for i in islands])

def rest_center(idxs):
    c = Vector()
    for i in idxs: c += me.vertices[i].co
    return c / len(idxs)

# group islands into 2 eyes by rest x
cs = [(rest_center(i), i) for i in islands]
xs = sorted(c.x for c, _ in cs)
split = (xs[0] + xs[-1]) / 2
groupA = [i for c, i in cs if c.x <= split]   # one eye (all its islands)
groupB = [i for c, i in cs if c.x >  split]
flatA = [x for i in groupA for x in i]; flatB = [x for i in groupB for x in i]
print("group sizes:", len(flatA), len(flatB))

# socket targets from body's rendered socket verts
awm = arm.matrix_world; bwm = body.matrix_world
restL = awm @ arm.data.bones["eye_L"].head_local
restR = awm @ arm.data.bones["eye_R"].head_local
sockL = [v.index for v in bme.vertices if (bwm @ v.co - restL).length < 0.025]
sockR = [v.index for v in bme.vertices if (bwm @ v.co - restR).length < 0.025]
def eval_center(ob, idxs):
    ev = ob.evaluated_get(dg); evme = ev.to_mesh(); mw = ev.matrix_world
    c = Vector()
    for i in idxs: c += mw @ evme.vertices[i].co
    c /= len(idxs); ev.to_mesh_clear(); return c
tL = eval_center(body, sockL); tR = eval_center(body, sockR)

# ball radius from largest island of group A (rest space ~ world scale 1)
big = max(groupA, key=len)
c0 = rest_center(big)
rad = max((me.vertices[i].co - c0).length for i in big)
print("ball radius ~", round(rad, 4))

# recess: into the head along +Y world (front is -Y)
recess = Vector((0, 1, 0)) * (rad * 0.55)

pairs = []
cA = eval_center(eyes, flatA); cB = eval_center(eyes, flatB)
if (cA - tL).length + (cB - tR).length <= (cA - tR).length + (cB - tL).length:
    pairs = [(flatA, tL), (flatB, tR)]
else:
    pairs = [(flatA, tR), (flatB, tL)]

mw3 = eyes.matrix_world.to_3x3().inverted()
for idxs, tgt in pairs:
    tgt = tgt + recess
    for it in range(5):
        cur = eval_center(eyes, idxs)
        d = tgt - cur
        if d.length < 4e-4: break
        dl = mw3 @ d
        for i in idxs: me.vertices[i].co += dl
        me.update(); dg.update()
    print("seated group at", [round(x,4) for x in eval_center(eyes, idxs)],
          "target", [round(x,4) for x in tgt])

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX5_DONE")

import bpy
from mathutils import Vector, Matrix

eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
arm  = bpy.data.objects["weaver_base2"]
me = eyes.data

# how many bones carry a non-identity pose? (context for safety)
posed = [pb.name for pb in arm.pose.bones if not all(
    abs(a-b) < 1e-5 for a,b in zip(sum([list(r) for r in pb.matrix_basis], []),
                                    sum([list(r) for r in Matrix.Identity(4)], [])))]
print("posed bones:", len(posed), posed[:8])

# clear pose ONLY on the eye bones
for nm in ("eye_L", "eye_R"):
    pb = arm.pose.bones[nm]
    pb.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()

dg = bpy.context.evaluated_depsgraph_get()
tL = arm.matrix_world @ arm.pose.bones["eye_L"].head
tR = arm.matrix_world @ arm.pose.bones["eye_R"].head
print("new pose targets L", [round(x,4) for x in tL], "R", [round(x,4) for x in tR])

mean_x = sum(v.co.x for v in me.vertices) / len(me.vertices)
clus = {"A": [v.index for v in me.vertices if v.co.x <= mean_x],
        "B": [v.index for v in me.vertices if v.co.x >  mean_x]}
def eval_center(idxs):
    ev = eyes.evaluated_get(dg); evme = ev.to_mesh(); mw = ev.matrix_world
    c = Vector()
    for i in idxs: c += mw @ evme.vertices[i].co
    c /= len(idxs); ev.to_mesh_clear(); return c
cA = eval_center(clus["A"]); cB = eval_center(clus["B"])
pairs = [("A", tL if (cA-tL).length < (cA-tR).length else tR),
         ("B", tL if (cB-tL).length < (cB-tR).length else tR)]
if pairs[0][1] == pairs[1][1]:  # both matched same target → force split
    pairs = [("A", tR), ("B", tL)] if cA.x > cB.x else [("A", tL), ("B", tR)]
mw3 = eyes.matrix_world.to_3x3().inverted()
for it in range(4):
    moved = 0.0
    for name, tgt in pairs:
        cur = eval_center(clus[name]); d = tgt - cur
        if d.length < 5e-4: continue
        dl = mw3 @ d
        for i in clus[name]: me.vertices[i].co += dl
        me.update(); dg.update(); moved = max(moved, d.length)
    if moved < 5e-4: break
for name, tgt in pairs:
    print(name, "final:", [round(x,4) for x in eval_center(clus[name])],
          "target:", [round(x,4) for x in tgt])
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX3_DONE")

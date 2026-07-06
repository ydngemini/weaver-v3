import bpy
from mathutils import Vector

eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
arm  = bpy.data.objects["weaver_base2"]
me = eyes.data
dg = bpy.context.evaluated_depsgraph_get()

# pose-space targets (what actually renders)
tL = arm.matrix_world @ arm.pose.bones["eye_L"].head
tR = arm.matrix_world @ arm.pose.bones["eye_R"].head
print("pose targets L", [round(x,4) for x in tL], "R", [round(x,4) for x in tR])
rL = arm.matrix_world @ arm.data.bones["eye_L"].head_local
print("rest target L", [round(x,4) for x in rL], "| pose==rest:", (tL-rL).length < 1e-5)

# split verts into two clusters by rest-x around mesh mean
mean_x = sum(v.co.x for v in me.vertices) / len(me.vertices)
clus = {"A": [v.index for v in me.vertices if v.co.x <= mean_x],
        "B": [v.index for v in me.vertices if v.co.x >  mean_x]}

def eval_center(idxs):
    ev = eyes.evaluated_get(dg)
    evme = ev.to_mesh()
    mw = ev.matrix_world
    c = Vector()
    for i in idxs: c += mw @ evme.vertices[i].co
    c /= len(idxs)
    ev.to_mesh_clear()
    return c

# assign clusters to nearer target
cA = eval_center(clus["A"]); cB = eval_center(clus["B"])
pairs = [("A", tL if (cA-tL).length < (cA-tR).length else tR),
         ("B", tL if (cB-tL).length < (cB-tR).length else tR)]
print("evaluated centers A", [round(x,4) for x in cA], "B", [round(x,4) for x in cB])

mw3 = eyes.matrix_world.to_3x3().inverted()
for it in range(4):
    moved = 0.0
    for name, tgt in pairs:
        idxs = clus[name]
        cur = eval_center(idxs)
        d = tgt - cur
        if d.length < 5e-4: continue
        dl = mw3 @ d
        for i in idxs: me.vertices[i].co += dl
        me.update()
        dg.update()
        moved = max(moved, d.length)
    print(f"iter{it} max residual {moved:.4f}")
    if moved < 5e-4: break

for name, tgt in pairs:
    print(name, "final:", [round(x,4) for x in eval_center(clus[name])],
          "target:", [round(x,4) for x in tgt])
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX2_DONE")

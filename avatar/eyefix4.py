import bpy
from mathutils import Vector

body = bpy.data.objects["weaver_base2-base"]
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
arm  = bpy.data.objects["weaver_base2"]
me, bme = eyes.data, body.data
dg = bpy.context.evaluated_depsgraph_get()

# socket vert sets: body verts whose REST world pos is near rest eye-bone heads
awm = arm.matrix_world; bwm = body.matrix_world
restL = awm @ arm.data.bones["eye_L"].head_local
restR = awm @ arm.data.bones["eye_R"].head_local
sockL = [v.index for v in bme.vertices if (bwm @ v.co - restL).length < 0.025]
sockR = [v.index for v in bme.vertices if (bwm @ v.co - restR).length < 0.025]
print("socket vert counts L/R:", len(sockL), len(sockR))

def eval_center(ob, idxs):
    ev = ob.evaluated_get(dg); evme = ev.to_mesh(); mw = ev.matrix_world
    c = Vector()
    for i in idxs: c += mw @ evme.vertices[i].co
    c /= len(idxs); ev.to_mesh_clear(); return c

tL = eval_center(body, sockL)   # where the sockets ACTUALLY render
tR = eval_center(body, sockR)
print("rendered socket centers L", [round(x,4) for x in tL], "R", [round(x,4) for x in tR])

mean_x = sum(v.co.x for v in me.vertices) / len(me.vertices)
clus = {"A": [v.index for v in me.vertices if v.co.x <= mean_x],
        "B": [v.index for v in me.vertices if v.co.x >  mean_x]}
cA = eval_center(eyes, clus["A"]); cB = eval_center(eyes, clus["B"])
pairs = [("A", tL if (cA-tL).length < (cA-tR).length else tR),
         ("B", tL if (cB-tL).length < (cB-tR).length else tR)]
if pairs[0][1] is pairs[1][1]:
    pairs = [("A", tL), ("B", tR)] if (cA-tL).length + (cB-tR).length <= (cA-tR).length + (cB-tL).length else [("A", tR), ("B", tL)]

mw3 = eyes.matrix_world.to_3x3().inverted()
for it in range(6):
    moved = 0.0
    for name, tgt in pairs:
        cur = eval_center(eyes, clus[name]); d = tgt - cur
        if d.length < 5e-4: continue
        # empirical: apply, measure gain, correct (handles skinned response)
        dl = mw3 @ d
        for i in clus[name]: me.vertices[i].co += dl
        me.update(); dg.update()
        moved = max(moved, d.length)
    if moved < 5e-4: break
for name, tgt in pairs:
    fin = eval_center(eyes, clus[name])
    print(name, "final:", [round(x,4) for x in fin], "target:", [round(x,4) for x in tgt],
          "residual:", round((fin-tgt).length, 5))

# restore eye material (was left green by diagnostic? that render wasn't saved — but be safe)
if eyes.data.materials and eyes.data.materials[0] and eyes.data.materials[0].name.startswith("G"):
    eb = bpy.data.materials.get("Eye_brown")
    eyes.data.materials.clear(); eyes.data.materials.append(eb)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX4_DONE")

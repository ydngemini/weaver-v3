import bpy
from mathutils import Vector

sc = bpy.context.scene
sc.frame_set(1)
body = bpy.data.objects["weaver_base2-base"]
arm = bpy.data.objects["weaver_base2"]
dg = bpy.context.evaluated_depsgraph_get()
gi = {g.name: g.index for g in body.vertex_groups}
def widx(names, thresh=0.4):
    out = []
    for v in body.data.vertices:
        s = 0.0
        for g in v.groups:
            for n in names:
                if g.group == gi[n]: s += g.weight
        if s > thresh: out.append(v.index)
    return out
def eval_center(idxs):
    ev = body.evaluated_get(dg); m = ev.to_mesh(); mw = ev.matrix_world
    c = Vector()
    for i in idxs: c += mw @ m.vertices[i].co
    c /= len(idxs); ev.to_mesh_clear(); return c

for label, names in (("hand_L", ["wrist_L"]), ("hand_R", ["wrist_R"]),
                     ("elbow_L", ["lowerarm01_L"]), ("shoulder_L", ["upperarm01_L"]),
                     ("head", ["head"])):
    idxs = widx(names)
    c = eval_center(idxs) if idxs else None
    print(f"{label:12s} evaluated: {[round(x,3) for x in c] if c else 'none'} ({len(idxs)} verts)")

awm = arm.matrix_world
for bn in ("upperarm01_L", "lowerarm01_L", "wrist_L"):
    b = arm.pose.bones[bn]
    print(f"bone {bn:14s} head: {[round(x,3) for x in (awm @ b.head)]} tail: {[round(x,3) for x in (awm @ b.tail)]}")

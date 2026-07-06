"""Skin the gold chains by nearest-body-vertex weight transfer + armature modifier."""
import bpy
from mathutils import kdtree

body = bpy.data.objects["weaver_base2-base"]
arm = bpy.data.objects["weaver_base2"]
bme = body.data

# KD-tree over body verts (both meshes share the same object space)
kd = kdtree.KDTree(len(bme.vertices))
for i, v in enumerate(bme.vertices):
    kd.insert(v.co, i)
kd.balance()
bnames = [g.name for g in body.vertex_groups]

for cname in ("belly_chain", "strap_chain"):
    ch = bpy.data.objects.get(cname)
    if not ch: continue
    # wipe existing groups, mirror body's group list
    for g in list(ch.vertex_groups): ch.vertex_groups.remove(g)
    groups = {nm: ch.vertex_groups.new(name=nm) for nm in bnames}
    for v in ch.data.vertices:
        co, idx, dist = kd.find(v.co)
        for g in bme.vertices[idx].groups:
            if g.weight > 0.01:
                groups[bnames[g.group]].add([v.index], g.weight, "REPLACE")
    if not any(m.type == 'ARMATURE' for m in ch.modifiers):
        am = ch.modifiers.new("Armature", "ARMATURE"); am.object = arm
        while ch.modifiers.find("Armature") > 0:
            bpy.context.view_layer.objects.active = ch
            bpy.ops.object.modifier_move_up(modifier="Armature")
    print("skinned:", cname, len(ch.data.vertices), "verts")

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("CHAINS_DONE")

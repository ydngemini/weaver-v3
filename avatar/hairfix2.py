import bpy
hair = bpy.data.objects["hair_braids"]
arm = bpy.data.objects["weaver_base2"]
vg = hair.vertex_groups.get("head") or hair.vertex_groups.new(name="head")
vg.add(list(range(len(hair.data.vertices))), 1.0, "REPLACE")
if not any(m.type == 'ARMATURE' for m in hair.modifiers):
    am = hair.modifiers.new("Armature", "ARMATURE"); am.object = arm
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("HAIRFIX2_DONE")

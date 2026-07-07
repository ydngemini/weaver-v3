"""Bake the posed deform into all skinned meshes, then apply pose as rest.
After this: rest pose == rendered pose — eyes/hair correct in ANY glTF viewer."""
import bpy

arm = bpy.data.objects["weaver_base2"]

# 1. bake current armature deform into every skinned mesh, re-add the modifier
skinned = [ob for ob in bpy.data.objects
           if ob.type == 'MESH' and any(m.type == 'ARMATURE' for m in ob.modifiers)]
for ob in skinned:
    bpy.context.view_layer.objects.active = ob
    mods = [m.name for m in ob.modifiers if m.type == 'ARMATURE']
    for mn in mods:
        bpy.ops.object.modifier_apply(modifier=mn)
    am = ob.modifiers.new("Armature", "ARMATURE"); am.object = arm
    # keep Armature first in the stack so Sub/Solid stay after it
    while ob.modifiers.find("Armature") > 0:
        bpy.ops.object.modifier_move_up(modifier="Armature")
    print("baked:", ob.name)

# 2. apply current pose as the new rest pose
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
bpy.ops.pose.select_all(action='SELECT')
bpy.ops.pose.armature_apply(selected=False)
bpy.ops.object.mode_set(mode='OBJECT')
print("pose applied as rest")

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("BIND_FIX_DONE")

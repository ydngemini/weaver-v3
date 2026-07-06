import bpy
from mathutils import Vector

sc = bpy.context.scene
sc.frame_set(1)
body = bpy.data.objects["weaver_base2-base"]
arm = bpy.data.objects["weaver_base2"]
dg = bpy.context.evaluated_depsgraph_get()

# ── ground her: evaluated feet min-z → move armature (children follow) ──
ev = body.evaluated_get(dg); m = ev.to_mesh(); mw = ev.matrix_world
minz = min((mw @ v.co).z for v in m.vertices)
ev.to_mesh_clear()
arm.location.z -= minz
arm.location.y -= 1.0          # step her toward the window
print(f"grounded: dropped {minz:.3f}, moved to window")

# ── extend ceiling over the backdrop gap ──
ceil = bpy.data.objects["ceiling"]
ceil.scale = (1.0, 1.85, 1.0); ceil.location.y = -2.4

# ── lighting boost ──
bpy.data.lights["key_warm"].energy = 450
bpy.data.lights["fill_cool"].energy = 170
wl = bpy.data.lights.new("window_rim", 'AREA'); wl.energy = 260; wl.color = (0.7, 0.8, 1.0); wl.size = 4.0
wo = bpy.data.objects.new("window_rim", wl); wo.location = (0, -3.2, 2.4)
wo.rotation_euler = (Vector((0, 1.2, -1.2))).to_track_quat('-Z', 'Y').to_euler()
bpy.data.collections["Apartment"].objects.link(wo)
bl = bpy.data.lights.new("bounce", 'AREA'); bl.energy = 110; bl.color = (1.0, 0.9, 0.8); bl.size = 5.0
bo = bpy.data.objects.new("bounce", bl); bo.location = (0, 0.5, 0.15)
bo.rotation_euler = (3.14159, 0, 0)
bpy.data.collections["Apartment"].objects.link(bo)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("APT_FIX_DONE")

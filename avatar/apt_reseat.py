import bpy
# positions live in mesh data (loc should be ~0); these are measured corrective deltas
fix = {
    "sofa_back": -0.30, "sofa_L_back": -0.30,
    "bed_mattress": -0.13, "bed_throw": -0.12,
    "pillow1": -0.23, "pillow2": -0.23,
    "island_top": 0.0, "bar_top": 0.0,
}
for nm, dz in fix.items():
    ob = bpy.data.objects.get(nm)
    if ob: ob.location = (0, 0, dz)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("RESEAT_DONE")

import bpy
# L-sofa extension collided with the bedroom zone — remove it (main sofa remains)
for nm in ("sofa_L_seat", "sofa_L_back", "headboard"):
    ob = bpy.data.objects.get(nm)
    if ob: bpy.data.objects.remove(ob, do_unlink=True)
# pillows onto the mattress (mesh-space deltas: measured pillow x≈-3.95, mattress spans -3.72..-2.68)
p1 = bpy.data.objects.get("pillow1"); p2 = bpy.data.objects.get("pillow2")
if p1: p1.location = (0.55, 0.15, -0.20)
if p2: p2.location = (0.55, -0.15, -0.20)
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("TIDY_DONE")

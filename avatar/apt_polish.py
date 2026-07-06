import bpy

sc = bpy.context.scene
# contact shading + a touch more ambient so furniture bodies read
try:
    sc.eevee.use_gtao = True
    sc.eevee.gtao_distance = 0.6
except AttributeError:
    pass  # EEVEE Next: AO via raytracing options
try:
    sc.eevee.use_raytracing = True
except AttributeError:
    pass
sc.world.node_tree.nodes["Background"].inputs[0].default_value = (0.018, 0.02, 0.03, 1)

# soft overall ceiling bounce
l = bpy.data.lights.new("ceiling_soft", 'AREA'); l.energy = 320; l.size = 7.0
l.color = (1.0, 0.92, 0.8)
lo = bpy.data.objects.new("ceiling_soft", l); lo.location = (0, 0.4, 2.95)
lo.rotation_euler = (0, 0, 0)
bpy.data.collections["Apartment"].objects.link(lo)

# close the visible gaps
fix = {
    "sofa_back":   ("z", 0.70), "sofa_L_back": ("z", 0.70),
    "bed_mattress":("z", 0.33), "bed_throw":  ("z", 0.34),
    "pillow1": ("z", 0.52), "pillow2": ("z", 0.52),
    "bar_top": ("z", 1.005), "island_top": ("z", 0.865),
}
for nm, (axis, val) in fix.items():
    ob = bpy.data.objects.get(nm)
    if ob: setattr(ob.location, axis, val)

# brighten the bedroom rug so the zone isn't a black pit
m = bpy.data.materials.get("BedRug")
if m: m.node_tree.nodes["Principled BSDF"].inputs["Base Color"].default_value = (0.38, 0.33, 0.27, 1)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("POLISH_DONE")

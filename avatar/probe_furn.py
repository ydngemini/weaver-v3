import bpy
from mathutils import Vector
dg = bpy.context.evaluated_depsgraph_get()
for nm in ("bed_platform","bed_mattress","pillow1","headboard","sofa_seat","sofa_back","island_top","bar_top"):
    ob = bpy.data.objects.get(nm)
    if not ob: print(nm, "MISSING"); continue
    ev = ob.evaluated_get(dg); m = ev.to_mesh(); mw = ev.matrix_world
    zs = [ (mw @ v.co).z for v in m.vertices ]
    ys = [ (mw @ v.co).y for v in m.vertices ]
    xs = [ (mw @ v.co).x for v in m.vertices ]
    ev.to_mesh_clear()
    print(f"{nm:14s} loc={[round(x,2) for x in ob.location]} scale={[round(x,2) for x in ob.scale]} "
          f"bboxX=({min(xs):.2f},{max(xs):.2f}) Y=({min(ys):.2f},{max(ys):.2f}) Z=({min(zs):.2f},{max(zs):.2f})")

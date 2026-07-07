import bpy
from mathutils import Vector
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
me = eyes.data
# pull eyes forward a touch (reduce recess from 0.55r to ~0.25r): -Y is forward
for v in me.vertices: v.co += eyes.matrix_world.to_3x3().inverted() @ Vector((0, -0.006, 0))
me.update()
# plain white material
wm = bpy.data.materials.new("EyeTest"); wm.use_nodes = True
bsdf = wm.node_tree.nodes["Principled BSDF"]
bsdf.inputs["Base Color"].default_value = (0.9, 0.9, 0.9, 1)
bsdf.inputs["Roughness"].default_value = 0.2
me.materials.clear(); me.materials.append(wm)
# UV sanity
uv = me.uv_layers.active
if uv:
    us = [d.uv[0] for d in uv.data[:200]]; vs = [d.uv[1] for d in uv.data[:200]]
    print("UV range sample: u", round(min(us),3), round(max(us),3), "v", round(min(vs),3), round(max(vs),3))
bpy.ops.wm.save_as_mainfile(filepath="/media/ydn/SYPHER_CORE2/weaver v3/avatar/eyetest.blend")
print("SAVED_TEST")

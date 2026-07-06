import bpy
g = bpy.data.objects.get("window_glass")
if g: g.hide_render = True; g.hide_viewport = True
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("GLASS_HIDDEN")

import bpy
import numpy as np

HAIR_DIR = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/assets/hair/hair02_ccby/hair/elvs_braided_rows/"

# ── rebake diffuse: contrast boost, then darken KEEPING strand detail ──
src = bpy.data.images.load(HAIR_DIR + "mh_cornrowstex1.png")
w, h = src.size
px = np.array(src.pixels[:], dtype=np.float32).reshape(h, w, 4)
rgb = px[..., :3]
rgb = np.clip((rgb - 0.5) * 1.45 + 0.5, 0, 1)          # contrast: pop the braid strands
lum = rgb.mean(axis=-1, keepdims=True)
rgb = lum * np.array([0.30, 0.26, 0.24]) + 0.015        # deep warm black, detail preserved
px[..., :3] = np.clip(rgb, 0, 1)
out = bpy.data.images.new("braids_black2", w, h, alpha=True)
out.pixels = px.ravel().tolist()
out.filepath_raw = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/textures/braids_black2.png"
out.file_format = 'PNG'; out.save()

# ── rebuild hair material: diffuse + NORMAL MAP (asset ships one) ──
nrm = bpy.data.images.load(HAIR_DIR + "normals.png")
nrm.colorspace_settings.name = 'Non-Color'
mat = bpy.data.materials["BraidsBlack"]
mat.use_nodes = True
nt = mat.node_tree; nt.nodes.clear()
outn = nt.nodes.new("ShaderNodeOutputMaterial")
b = nt.nodes.new("ShaderNodeBsdfPrincipled")
t = nt.nodes.new("ShaderNodeTexImage"); t.image = out
n = nt.nodes.new("ShaderNodeTexImage"); n.image = nrm
nm = nt.nodes.new("ShaderNodeNormalMap"); nm.inputs["Strength"].default_value = 1.4
b.inputs["Roughness"].default_value = 0.42
nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
nt.links.new(n.outputs["Color"], nm.inputs["Color"])
nt.links.new(nm.outputs["Normal"], b.inputs["Normal"])
nt.links.new(b.outputs["BSDF"], outn.inputs["Surface"])
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("HAIRTEX_DONE")

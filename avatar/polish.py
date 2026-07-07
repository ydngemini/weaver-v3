import bpy, bmesh
import numpy as np
from mathutils import Vector

# ── 1. bake darkened textures to files (glTF-compatible simple graphs) ──
def bake_mult(img_name_contains, mult, outpath):
    src = None
    for img in bpy.data.images:
        if img_name_contains in img.name.lower() and img.size[0] > 0:
            src = img; break
    if src is None:
        print("MISS", img_name_contains); return None
    w, h = src.size
    px = np.array(src.pixels[:], dtype=np.float32).reshape(h, w, 4)
    px[..., :3] *= np.array(mult, dtype=np.float32)
    out = bpy.data.images.new(outpath.split("/")[-1], w, h, alpha=True)
    out.pixels = px.ravel().tolist()
    out.filepath_raw = outpath; out.file_format = 'PNG'
    out.save()
    return out

skin_img = bake_mult("darkskinned", (1-0.55*(1-0.42), 1-0.55*(1-0.33), 1-0.55*(1-0.30)),
                     "/media/ydn/SYPHER_CORE2/weaver v3/avatar/textures/skin_dark.png")
hair_img = bake_mult("cornrow", (0.9*0.06+0.1, 0.9*0.05+0.1, 0.9*0.05+0.1),
                     "/media/ydn/SYPHER_CORE2/weaver v3/avatar/textures/braids_black.png")

def simple_graph(mat, img, rough):
    mat.use_nodes = True
    nt = mat.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial")
    b = nt.nodes.new("ShaderNodeBsdfPrincipled")
    t = nt.nodes.new("ShaderNodeTexImage"); t.image = img
    b.inputs["Roughness"].default_value = rough
    nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
    nt.links.new(b.outputs["BSDF"], out.inputs["Surface"])

if skin_img: simple_graph(bpy.data.materials["DefaultSkin"], skin_img, 0.45)
if hair_img: simple_graph(bpy.data.materials["BraidsBlack"], hair_img, 0.6)
print("textures baked")

# ── 2. suit boundary relax (kill remaining raggedness) ──
suit = bpy.data.objects.get("monokini")
if suit:
    bm = bmesh.new(); bm.from_mesh(suit.data); bm.verts.ensure_lookup_table()
    boundary = [v for v in bm.verts if any(e.is_boundary for e in v.link_edges)]
    for _ in range(5):
        upd = {}
        for v in boundary:
            nbr = [e.other_vert(v) for e in v.link_edges]
            if nbr: upd[v] = sum((n.co for n in nbr), Vector()) / len(nbr)
        for v, c in upd.items(): v.co = v.co * 0.45 + c * 0.55
    bm.to_mesh(suit.data); bm.free()
    print("suit relaxed")

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("POLISH_DONE")

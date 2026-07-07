import bpy, bmesh
from mathutils import Vector

eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
me = eyes.data
uvl = me.uv_layers.active

# island detection with per-island UV centroid
bm = bmesh.new(); bm.from_mesh(me)
bm.verts.ensure_lookup_table(); bm.faces.ensure_lookup_table()
uv_layer = bm.loops.layers.uv.active
seen, islands = set(), []
for v in bm.verts:
    if v.index in seen: continue
    stack, isl = [v], set()
    while stack:
        cur = stack.pop()
        if cur.index in isl: continue
        isl.add(cur.index)
        for e in cur.link_edges:
            o = e.other_vert(cur)
            if o.index not in isl: stack.append(o)
    seen |= isl; islands.append(isl)

# classify islands: cornea if UV centroid in the small lavender disc (u>0.8, v<0.25)
cornea_verts = set()
for isl in islands:
    us, vs, n = 0.0, 0.0, 0
    for f in bm.faces:
        for loop in f.loops:
            if loop.vert.index in isl:
                us += loop[uv_layer].uv.x; vs += loop[uv_layer].uv.y; n += 1
    if n and us/n > 0.8 and vs/n < 0.25:
        cornea_verts |= isl
print("cornea verts to delete:", len(cornea_verts))
bmesh.ops.delete(bm, geom=[v for v in bm.verts if v.index in cornea_verts], context="VERTS")
bm.to_mesh(me); bm.free()
me.update()

# per-ball: scale 0.92 around own center + extra recess
bm = bmesh.new(); bm.from_mesh(me); bm.verts.ensure_lookup_table()
seen, islands = set(), []
for v in bm.verts:
    if v.index in seen: continue
    stack, isl = [v], set()
    while stack:
        cur = stack.pop()
        if cur.index in isl: continue
        isl.add(cur.index)
        for e in cur.link_edges:
            o = e.other_vert(cur)
            if o.index not in isl: stack.append(o)
    seen |= isl; islands.append(sorted(isl))
bm.free()
mw3 = eyes.matrix_world.to_3x3().inverted()
recess = mw3 @ Vector((0, 0.010, 0))
for isl in islands:
    c = Vector()
    for i in isl: c += me.vertices[i].co
    c /= len(isl)
    for i in isl:
        me.vertices[i].co = c + (me.vertices[i].co - c) * 0.92 + recess
me.update()
print("balls remaining:", [len(i) for i in islands])

# restore textured eye material
eb = bpy.data.materials.get("Eye_brown")
eb.use_nodes = True
nt = eb.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
tex = nt.nodes.new("ShaderNodeTexImage")
tex.image = bpy.data.images.get("brown_eye_png") or bpy.data.images.load("/home/ydn/Documents/makehuman/v1py3/exports/textures/brown_eye.png")
bsdf.inputs["Roughness"].default_value = 0.15
nt.links.new(tex.outputs["Color"], bsdf.inputs["Base Color"])
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
me.materials.clear(); me.materials.append(eb)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX7_DONE")

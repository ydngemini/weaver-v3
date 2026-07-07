import bpy, json
out={}
for ob in bpy.data.objects:
    e={'type':ob.type,'hide':ob.hide_viewport}
    if ob.type=='MESH':
        m=ob.data
        e.update(verts=len(m.vertices),polys=len(m.polygons),
                 uvs=[u.name for u in m.uv_layers],
                 vgroups=len(ob.vertex_groups),
                 vgroup_names=[g.name for g in ob.vertex_groups][:200],
                 materials=[ms.material.name if ms.material else None for ms in ob.material_slots],
                 dims=[round(d,3) for d in ob.dimensions])
    if ob.type=='ARMATURE':
        e.update(bones=len(ob.data.bones))
    out[ob.name]=e
print("SCENE_JSON="+json.dumps(out))

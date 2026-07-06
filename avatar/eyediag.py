import bpy
from mathutils import Vector
arm = bpy.data.objects["weaver_base2"]
eyes = bpy.data.objects["weaver_base2-highpolyeyes"]

# report eyes' modifier/parent situation — the missing fact
print("EYES parent:", eyes.parent.name, "| parent_type:", eyes.parent_type)
print("EYES modifiers:", [(m.type, getattr(m, 'object', None) and m.object.name) for m in eyes.modifiers])
body = bpy.data.objects["weaver_base2-base"]
print("BODY modifiers:", [(m.type, getattr(m, 'object', None) and m.object.name) for m in body.modifiers])

def marker(name, loc, color):
    mesh = bpy.data.meshes.new(name)
    import bmesh
    bm = bmesh.new(); bmesh.ops.create_uvsphere(bm, u_segments=8, v_segments=6, radius=0.012)
    bm.to_mesh(mesh); bm.free()
    ob = bpy.data.objects.new(name, mesh)
    bpy.context.scene.collection.objects.link(ob)
    ob.location = loc
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; nt.nodes.clear()
    out = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
    em.inputs["Color"].default_value = color; em.inputs["Strength"].default_value = 5
    nt.links.new(em.outputs[0], out.inputs[0]); mesh.materials.append(m)
    return ob

for nm, col in (("eye_L", (1,0,0,1)), ("eye_R", (1,0,0,1))):          # RED = rest
    marker("rest_"+nm, arm.matrix_world @ arm.data.bones[nm].head_local, col)
for nm, col in (("eye_L", (0,0.3,1,1)), ("eye_R", (0,0.3,1,1))):      # BLUE = pose
    marker("pose_"+nm, arm.matrix_world @ arm.pose.bones[nm].head, col)

# eyes → emissive green
gm = bpy.data.materials.new("G"); gm.use_nodes = True
nt = gm.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
em.inputs["Color"].default_value = (0,1,0,1); em.inputs["Strength"].default_value = 5
nt.links.new(em.outputs[0], out.inputs[0])
eyes.data.materials.clear(); eyes.data.materials.append(gm)

# render head
sc = bpy.context.scene
sc.render.engine = "BLENDER_EEVEE_NEXT"; sc.render.resolution_x = sc.render.resolution_y = 1024
cam = bpy.data.objects["PrevCam"]
mwb = body.matrix_world
corners = [mwb @ Vector(c) for c in body.bound_box]
cmin = Vector((min(c[i] for c in corners) for i in range(3)))
cmax = Vector((max(c[i] for c in corners) for i in range(3)))
center = (cmin+cmax)/2; height = (cmax-cmin).z
head_t = center + Vector((0,0,1))*height*0.405
cam.location = head_t + Vector((0,-1,0))*(height*0.35)
cam.rotation_euler = (head_t - cam.location).to_track_quat("-Z","Y").to_euler()
sc.render.filepath = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/renders/eyediag.png"
bpy.ops.render.render(write_still=True)
print("DIAG_RENDERED")   # NOTE: not saving the blend — markers are throwaway

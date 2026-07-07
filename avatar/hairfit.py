import bpy, math
from mathutils import Vector

HAIR_OBJ = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/assets/hair/hair02_ccby/hair/elvs_braided_rows/cornrowsofelv5.obj"
HAIR_TEX = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/assets/hair/hair02_ccby/hair/elvs_braided_rows/mh_cornrowstex1.png"

body = bpy.data.objects["weaver_base2-base"]
me = body.data
gi = {g.name: g.index for g in body.vertex_groups}
def w(v, *names):
    s = 0.0
    for g in v.groups:
        for n in names:
            if n in gi and g.group == gi[n]: s += g.weight
    return min(s, 1.0)

# head bbox in WORLD (rest ~ render here)
bwm = body.matrix_world
hv = [bwm @ v.co for v in me.vertices if w(v, "head") > 0.5]
hmin = Vector((min(p[i] for p in hv) for i in range(3)))
hmax = Vector((max(p[i] for p in hv) for i in range(3)))
hc = (hmin + hmax) / 2
head_w = hmax.x - hmin.x
print("head bbox w:", round(head_w,3), "top z:", round(hmax.z,3), "center:", [round(x,3) for x in hc])

bpy.ops.wm.obj_import(filepath=HAIR_OBJ)
hair = bpy.context.selected_objects[0]
hair.name = "hair_braids"
bb = [hair.matrix_world @ Vector(c) for c in hair.bound_box]
bmin = Vector((min(c[i] for c in bb) for i in range(3)))
bmax = Vector((max(c[i] for c in bb) for i in range(3)))
print("hair raw bbox min:", [round(x,2) for x in bmin], "max:", [round(x,2) for x in bmax])

# scale: hair skull width ≈ head width * 1.16  (hair x extent is the skull cap width)
s = head_w * 1.16 / (bmax.x - bmin.x)
hair.scale = (s, s, s)
bpy.context.view_layer.update()
bb = [hair.matrix_world @ Vector(c) for c in hair.bound_box]
bmin = Vector((min(c[i] for c in bb) for i in range(3)))
bmax = Vector((max(c[i] for c in bb) for i in range(3)))
# position: top of hair slightly above head top, centered on head x/y
delta = Vector((hc.x - (bmin.x+bmax.x)/2,
                hc.y - (bmin.y+bmax.y)/2,
                (hmax.z + 0.012) - bmax.z))
hair.location += delta
bpy.context.view_layer.update()
print("hair placed, scale", round(s,4))

# black-tinted material
mat = bpy.data.materials.new("BraidsBlack"); mat.use_nodes = True
nt = mat.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial")
bsdf = nt.nodes.new("ShaderNodeBsdfPrincipled")
tex = nt.nodes.new("ShaderNodeTexImage"); tex.image = bpy.data.images.load(HAIR_TEX)
mix = nt.nodes.new("ShaderNodeMix"); mix.data_type='RGBA'; mix.blend_type='MULTIPLY'
mix.inputs["Factor"].default_value = 0.9
mix.inputs[7].default_value = (0.06, 0.05, 0.05, 1)   # near-black
nt.links.new(tex.outputs["Color"], mix.inputs[6])
nt.links.new(mix.outputs[2], bsdf.inputs["Base Color"])
bsdf.inputs["Roughness"].default_value = 0.6
nt.links.new(bsdf.outputs["BSDF"], out.inputs["Surface"])
hair.data.materials.clear(); hair.data.materials.append(mat)

# parent to head bone so it follows the rig
arm = bpy.data.objects["weaver_base2"]
hair.parent = arm; hair.parent_type = 'BONE'; hair.parent_bone = 'head'
# keep world transform
hair.matrix_parent_inverse = (arm.matrix_world @ arm.pose.bones['head'].matrix).inverted()

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("HAIR_DONE")

import bpy
from mathutils import Vector

eyes = bpy.data.objects["weaver_base2-highpolyeyes"]
arm  = bpy.data.objects["weaver_base2"]
target = (arm.matrix_world @ arm.data.bones["eye_L"].head_local +
          arm.matrix_world @ arm.data.bones["eye_R"].head_local) / 2

def world_center():
    mw = eyes.matrix_world
    c = Vector()
    for v in eyes.data.vertices:
        c += mw @ v.co
    return c / len(eyes.data.vertices)

for it in range(3):                      # iterate: measure → move → re-measure
    cur = world_center()
    d_world = target - cur
    if d_world.length < 1e-4:
        break
    probe = eyes.data.vertices[0].co.copy()
    # empirical local direction: move one unit-ish, measure world response
    mw = eyes.matrix_world
    d_local_guess = mw.to_3x3().inverted() @ d_world
    for v in eyes.data.vertices:
        v.co += d_local_guess
    eyes.data.update()
    ach = world_center() - cur
    err = d_world - ach
    print(f"iter{it}: wanted {[round(x,4) for x in d_world]} achieved {[round(x,4) for x in ach]}")
print("final center:", [round(x,4) for x in world_center()], "target:", [round(x,4) for x in target])

# eye material sanity: report nodes + image status
mat = None
for ms in eyes.material_slots:
    if ms.material: mat = ms.material
print("eye mat:", mat.name if mat else None, "| use_nodes:", mat.use_nodes if mat else None)
if mat and mat.use_nodes:
    for n in mat.node_tree.nodes:
        if n.type == 'TEX_IMAGE':
            img = n.image
            print("  tex node:", img.name if img else None,
                  "| size:", (img.size[0], img.size[1]) if img else None,
                  "| filepath:", img.filepath if img else None)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("EYEFIX_DONE")

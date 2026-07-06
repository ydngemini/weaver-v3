import bpy, math
from mathutils import Vector, Matrix
hair = bpy.data.objects["hair_braids"]
bb = [hair.matrix_world @ Vector(c) for c in hair.bound_box]
c = sum(bb, Vector()) / 8
M = Matrix.Translation(c) @ Matrix.Rotation(math.pi, 4, 'Z') @ Matrix.Translation(-c)
hair.matrix_world = M @ hair.matrix_world
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("ROTATED")

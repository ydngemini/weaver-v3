import bpy, sys
from mathutils import Vector, Matrix
args = sys.argv[sys.argv.index("--")+1:] if "--" in sys.argv else ["0.05", "0.015", "1.10"]
dz, dy, s = float(args[0]), float(args[1]), float(args[2])
hair = bpy.data.objects["hair_braids"]
bb = [hair.matrix_world @ Vector(c) for c in hair.bound_box]
c = sum(bb, Vector()) / 8
M = Matrix.Translation(Vector((0, dy, dz))) @ Matrix.Translation(c) @ Matrix.Scale(s, 4) @ Matrix.Translation(-c)
hair.matrix_world = M @ hair.matrix_world
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("NUDGED", dz, dy, s)

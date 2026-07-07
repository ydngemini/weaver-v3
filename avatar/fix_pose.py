"""Reset pose; greedy empirical arm-lowering (immune to axis/chirality surprises); re-key idle."""
import bpy, math
from mathutils import Vector, Quaternion

arm = bpy.data.objects["weaver_arm" if "weaver_arm" in bpy.data.objects else "weaver_base2"]
if arm.animation_data: arm.animation_data_clear()
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones
awm3 = arm.matrix_world.to_3x3()

# full reset to (baked) rest
for b in pb:
    b.rotation_quaternion = Quaternion()
    b.location = (0, 0, 0); b.scale = (1, 1, 1)
bpy.context.view_layer.update()

def wdir(name):
    b = pb[name]
    return (awm3 @ (b.tail - b.head)).normalized()

def greedy_aim(name, target, max_iter=40, step_deg=6.0):
    tgt = Vector(target).normalized()
    b = pb[name]
    for it in range(max_iter):
        cur = wdir(name)
        err = cur.angle(tgt)
        if err < math.radians(6): break
        best, bestq = err, None
        for axis in ((1,0,0), (0,1,0), (0,0,1)):
            for sgn in (1, -1):
                q0 = b.rotation_quaternion.copy()
                b.rotation_quaternion = q0 @ Quaternion(axis, math.radians(step_deg * sgn))
                bpy.context.view_layer.update()
                e = wdir(name).angle(tgt)
                if e < best: best, bestq = e, b.rotation_quaternion.copy()
                b.rotation_quaternion = q0
        bpy.context.view_layer.update()
        if bestq is None:
            step_deg *= 0.5
            if step_deg < 1.0: break
            continue
        b.rotation_quaternion = bestq
        bpy.context.view_layer.update()
    print(f"{name}: err {math.degrees(wdir(name).angle(tgt)):.1f}°")

greedy_aim("upperarm01_L", ( 0.30, -0.06, -0.95))
greedy_aim("upperarm01_R", (-0.30, -0.06, -0.95))
greedy_aim("lowerarm01_L", ( 0.24, -0.20, -0.95))
greedy_aim("lowerarm01_R", (-0.24, -0.20, -0.95))
greedy_aim("wrist_L", ( 0.20, -0.18, -0.96))
greedy_aim("wrist_R", (-0.20, -0.18, -0.96))

# idle loop (subtle — safe regardless of local-axis semantics)
sc = bpy.context.scene
sc.frame_start = 1; sc.frame_end = 120; sc.render.fps = 24
IDLE = ["spine01", "spine02", "spine03", "neck01", "head",
        "clavicle_L", "clavicle_R", "upperarm01_L", "upperarm01_R", "breast_L", "breast_R"]
def key_all(frame):
    for n in IDLE: pb[n].keyframe_insert("rotation_quaternion", frame=frame)
base = {n: pb[n].rotation_quaternion.copy() for n in IDLE}
key_all(1)
def rot_local(name, axis, deg):
    pb[name].rotation_quaternion = pb[name].rotation_quaternion @ Quaternion(axis, math.radians(deg))
rot_local("spine02", (1,0,0), -2.0); rot_local("spine01", (1,0,0), -1.2)
rot_local("spine03", (0,0,1), 1.0);  rot_local("neck01", (1,0,0), 1.0)
rot_local("head", (0,1,0), 3.0)
rot_local("clavicle_L", (1,0,0), -1.4); rot_local("clavicle_R", (1,0,0), -1.4)
rot_local("upperarm01_L", (0,0,1), 1.0); rot_local("upperarm01_R", (0,0,1), -1.0)
key_all(60)
for n in IDLE: pb[n].rotation_quaternion = base[n]
key_all(120)
act = arm.animation_data.action; act.name = "idle"
for fc in act.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'; kp.easing = 'EASE_IN_OUT'

bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("FIX_POSE_DONE")

"""Pose fix for point-joint (DAE) rigs: limb dir = child joint − parent joint."""
import bpy, math
from mathutils import Vector, Quaternion, Matrix

arm = bpy.data.objects["weaver_base2"]
if arm.animation_data: arm.animation_data_clear()
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones
awm3 = arm.matrix_world.to_3x3()
print("rotation_mode sample:", pb["upperarm01_L"].rotation_mode)

# hard reset — mode-agnostic
for b in pb: b.matrix_basis = Matrix.Identity(4)
bpy.context.view_layer.update()

CHAIN = {  # bone → the child joint that defines its limb direction
    "upperarm01_L": "lowerarm01_L", "upperarm01_R": "lowerarm01_R",
    "lowerarm01_L": "wrist_L",      "lowerarm01_R": "wrist_R",
    "wrist_L": "metacarpal3_L",     "wrist_R": "metacarpal3_R",
}
def limb_dir(name):
    return (awm3 @ (pb[CHAIN[name]].head - pb[name].head)).normalized()

def greedy_aim(name, target, max_iter=60, step_deg=7.0):
    tgt = Vector(target).normalized()
    b = pb[name]
    for it in range(max_iter):
        err = limb_dir(name).angle(tgt)
        if err < math.radians(5): break
        best, bestm = err, None
        for axis in ((1,0,0),(0,1,0),(0,0,1)):
            for sgn in (1,-1):
                m0 = b.matrix_basis.copy()
                b.matrix_basis = m0 @ Matrix.Rotation(math.radians(step_deg*sgn), 4, axis)
                bpy.context.view_layer.update()
                e = limb_dir(name).angle(tgt)
                if e < best: best, bestm = e, b.matrix_basis.copy()
                b.matrix_basis = m0
        bpy.context.view_layer.update()
        if bestm is None:
            step_deg *= 0.5
            if step_deg < 0.8: break
            continue
        b.matrix_basis = bestm; bpy.context.view_layer.update()
    print(f"{name}: rest_dir_now {[round(x,2) for x in limb_dir(name)]} err {math.degrees(limb_dir(name).angle(tgt)):.1f}°")

print("rest upperarm_L dir:", [round(x,2) for x in limb_dir("upperarm01_L")])
greedy_aim("upperarm01_L", ( 0.30, -0.05, -0.95))
greedy_aim("upperarm01_R", (-0.30, -0.05, -0.95))
greedy_aim("lowerarm01_L", ( 0.24, -0.18, -0.95))
greedy_aim("lowerarm01_R", (-0.24, -0.18, -0.95))
greedy_aim("wrist_L", ( 0.22, -0.16, -0.96))
greedy_aim("wrist_R", (-0.22, -0.16, -0.96))

# idle loop — keyframe matrix_basis-derived channels per rotation mode
sc = bpy.context.scene
sc.frame_start = 1; sc.frame_end = 160; sc.render.fps = 24
IDLE = ["root","pelvis_L","pelvis_R",
        "spine01","spine02","spine03","neck01","head","clavicle_L","clavicle_R",
        "upperarm01_L","upperarm01_R","lowerarm01_L","lowerarm01_R","wrist_L","wrist_R",
        "upperleg01_L","upperleg01_R","lowerleg01_L","lowerleg01_R","foot_L","foot_R","toe1-1_L","toe1-1_R",
        "breast_L","breast_R"]
IDLE = [n for n in IDLE if n in pb]
def key_all(frame):
    for n in IDLE:
        ch = "rotation_quaternion" if pb[n].rotation_mode == 'QUATERNION' else "rotation_euler"
        pb[n].keyframe_insert(ch, frame=frame)
base = {n: pb[n].matrix_basis.copy() for n in IDLE}
key_all(1)
def rot_local(name, axis, deg):
    if name in pb:
        pb[name].matrix_basis = pb[name].matrix_basis @ Matrix.Rotation(math.radians(deg), 4, axis)
def restore_base():
    for n, m in base.items():
        pb[n].matrix_basis = m.copy()
    bpy.context.view_layer.update()
def phase(frame, edits):
    restore_base()
    for name, axis, deg in edits:
        rot_local(name, axis, deg)
    bpy.context.view_layer.update()
    key_all(frame)
phase(36, [
    ("root",(0,1,0),1.6), ("root",(0,0,1),-2.0),
    ("pelvis_L",(0,0,1),-1.4), ("pelvis_R",(0,0,1),1.4),
    ("spine01",(1,0,0),-0.9), ("spine02",(1,0,0),-1.9),
    ("spine03",(0,0,1),0.8),  ("neck01",(1,0,0),0.7),
    ("head",(0,1,0),2.2),     ("head",(1,0,0),-0.5),
    ("clavicle_L",(1,0,0),-1.0), ("clavicle_R",(1,0,0),-1.0),
    ("upperarm01_L",(0,0,1),0.8), ("upperarm01_R",(0,0,1),-0.8),
    ("wrist_L",(0,1,0),0.9), ("wrist_R",(0,1,0),-0.7),
    ("upperleg01_L",(1,0,0),-2.8), ("lowerleg01_L",(1,0,0),4.0),
    ("foot_L",(1,0,0),-2.2), ("toe1-1_R",(1,0,0),1.6),
])
phase(76, [
    ("root",(0,1,0),-2.2), ("root",(0,0,1),2.4),
    ("pelvis_L",(0,0,1),1.7), ("pelvis_R",(0,0,1),-1.7),
    ("spine01",(1,0,0),-1.4), ("spine02",(1,0,0),-2.4),
    ("spine03",(0,0,1),-0.7), ("neck01",(1,0,0),1.0),
    ("head",(0,1,0),-2.8),    ("head",(0,0,1),0.5),
    ("clavicle_L",(1,0,0),-1.5), ("clavicle_R",(1,0,0),-1.3),
    ("upperarm01_L",(0,0,1),1.0), ("upperarm01_R",(0,0,1),-1.1),
    ("lowerarm01_L",(1,0,0),0.8), ("lowerarm01_R",(1,0,0),1.0),
    ("upperleg01_R",(1,0,0),-3.2), ("lowerleg01_R",(1,0,0),4.4),
    ("foot_R",(1,0,0),-2.4), ("toe1-1_L",(1,0,0),1.4),
])
phase(118, [
    ("root",(0,1,0),1.0), ("root",(0,0,1),-1.0),
    ("pelvis_L",(0,0,1),-0.8), ("pelvis_R",(0,0,1),0.8),
    ("spine01",(1,0,0),-0.5), ("spine02",(1,0,0),-1.2),
    ("spine03",(0,0,1),0.45), ("neck01",(1,0,0),0.4),
    ("head",(0,1,0),1.1),     ("head",(1,0,0),0.35),
    ("clavicle_L",(0,0,1),0.5), ("clavicle_R",(0,0,1),-0.5),
    ("wrist_L",(0,0,1),0.6), ("wrist_R",(0,0,1),-0.6),
    ("upperleg01_L",(1,0,0),-1.0), ("upperleg01_R",(1,0,0),1.0),
    ("foot_L",(1,0,0),-0.7), ("foot_R",(1,0,0),0.7),
])
restore_base()
key_all(160)
act = arm.animation_data.action; act.name = "idle"
for fc in act.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'; kp.easing = 'EASE_IN_OUT'
bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("FIX_POSE2_DONE")

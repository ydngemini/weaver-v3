"""Relaxed base pose + 160-frame looping idle (breath, sway, head, hands)."""
import bpy, math
from mathutils import Vector, Matrix, Quaternion

arm = bpy.data.objects["weaver_base2"]
bpy.context.view_layer.objects.active = arm
bpy.ops.object.mode_set(mode='POSE')
pb = arm.pose.bones
awm = arm.matrix_world
awm_inv = awm.inverted()

def aim_bone(name, target_world_dir, factor=1.0):
    """Rotate pose bone so its direction moves toward target (world), by factor."""
    b = pb[name]
    cur = (awm.to_3x3() @ (b.tail - b.head)).normalized()
    tgt = Vector(target_world_dir).normalized()
    q = cur.rotation_difference(tgt)
    q = Quaternion().slerp(q, factor)
    Rw = q.to_matrix().to_4x4()
    Ra = awm_inv @ Rw @ awm                      # world → armature space
    head = b.head.copy()
    b.matrix = Matrix.Translation(head) @ Ra.to_3x3().to_4x4() @ Matrix.Translation(-head) @ b.matrix
    bpy.context.view_layer.update()

# ── relaxed stance: arms down (~18° from vertical), slight forward, elbows soft ──
aim_bone("upperarm01_L", ( 0.30, -0.10, -0.95))
aim_bone("upperarm01_R", (-0.30, -0.10, -0.95))
aim_bone("lowerarm01_L", ( 0.22, -0.20, -0.95))
aim_bone("lowerarm01_R", (-0.22, -0.20, -0.95))
aim_bone("wrist_L", ( 0.18, -0.15, -0.97))
aim_bone("wrist_R", (-0.18, -0.15, -0.97))

# ── idle keyframes ──
sc = bpy.context.scene
sc.frame_start = 1; sc.frame_end = 160; sc.render.fps = 24
IDLE = ["root", "pelvis_L", "pelvis_R",
        "spine01", "spine02", "spine03", "neck01", "head",
        "clavicle_L", "clavicle_R", "upperarm01_L", "upperarm01_R",
        "lowerarm01_L", "lowerarm01_R", "wrist_L", "wrist_R",
        "upperleg01_L", "upperleg01_R", "lowerleg01_L", "lowerleg01_R",
        "foot_L", "foot_R", "toe1-1_L", "toe1-1_R",
        "breast_L", "breast_R"]
IDLE = [n for n in IDLE if n in pb]

def key_all(frame):
    for n in IDLE:
        ch = "rotation_quaternion" if pb[n].rotation_mode == 'QUATERNION' else "rotation_euler"
        pb[n].keyframe_insert(ch, frame=frame)

def rot_local(name, axis, deg):
    if name not in pb:
        return
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

# frame 1: base pose
base = {n: pb[n].matrix_basis.copy() for n in IDLE}
key_all(1)

# inhale, glance, settle, counter-sway, return. The asymmetry keeps her from
# reading as a mechanical metronome once the clip loops in glTF.
phase(36, [
    ("root", (0, 1, 0), 1.6), ("root", (0, 0, 1), -2.0),
    ("pelvis_L", (0, 0, 1), -1.4), ("pelvis_R", (0, 0, 1), 1.4),
    ("spine01", (1, 0, 0), -0.9), ("spine02", (1, 0, 0), -1.9),
    ("spine03", (0, 0, 1),  0.8), ("neck01",  (1, 0, 0),  0.7),
    ("head",    (0, 1, 0),  2.2), ("head",    (1, 0, 0), -0.5),
    ("clavicle_L", (1, 0, 0), -1.0), ("clavicle_R", (1, 0, 0), -1.0),
    ("upperarm01_L", (0, 0, 1),  0.8), ("upperarm01_R", (0, 0, 1), -0.8),
    ("wrist_L", (0, 1, 0), 0.9), ("wrist_R", (0, 1, 0), -0.7),
    ("upperleg01_L", (1, 0, 0), -2.8), ("lowerleg01_L", (1, 0, 0), 4.0),
    ("foot_L", (1, 0, 0), -2.2), ("toe1-1_R", (1, 0, 0), 1.6),
])
phase(76, [
    ("root", (0, 1, 0), -2.2), ("root", (0, 0, 1), 2.4),
    ("pelvis_L", (0, 0, 1), 1.7), ("pelvis_R", (0, 0, 1), -1.7),
    ("spine01", (1, 0, 0), -1.4), ("spine02", (1, 0, 0), -2.4),
    ("spine03", (0, 0, 1), -0.7), ("neck01",  (1, 0, 0),  1.0),
    ("head",    (0, 1, 0), -2.8), ("head",    (0, 0, 1),  0.5),
    ("clavicle_L", (1, 0, 0), -1.5), ("clavicle_R", (1, 0, 0), -1.3),
    ("upperarm01_L", (0, 0, 1),  1.0), ("upperarm01_R", (0, 0, 1), -1.1),
    ("lowerarm01_L", (1, 0, 0), 0.8), ("lowerarm01_R", (1, 0, 0), 1.0),
    ("upperleg01_R", (1, 0, 0), -3.2), ("lowerleg01_R", (1, 0, 0), 4.4),
    ("foot_R", (1, 0, 0), -2.4), ("toe1-1_L", (1, 0, 0), 1.4),
])
phase(118, [
    ("root", (0, 1, 0), 1.0), ("root", (0, 0, 1), -1.0),
    ("pelvis_L", (0, 0, 1), -0.8), ("pelvis_R", (0, 0, 1), 0.8),
    ("spine01", (1, 0, 0), -0.5), ("spine02", (1, 0, 0), -1.2),
    ("spine03", (0, 0, 1),  0.45), ("neck01", (1, 0, 0), 0.4),
    ("head",    (0, 1, 0),  1.1), ("head",   (1, 0, 0), 0.35),
    ("clavicle_L", (0, 0, 1), 0.5), ("clavicle_R", (0, 0, 1), -0.5),
    ("wrist_L", (0, 0, 1), 0.6), ("wrist_R", (0, 0, 1), -0.6),
    ("upperleg01_L", (1, 0, 0), -1.0), ("upperleg01_R", (1, 0, 0), 1.0),
    ("foot_L", (1, 0, 0), -0.7), ("foot_R", (1, 0, 0), 0.7),
])
restore_base()
key_all(160)

# smooth interpolation
act = arm.animation_data.action
for fc in act.fcurves:
    for kp in fc.keyframe_points:
        kp.interpolation = 'BEZIER'; kp.easing = 'EASE_IN_OUT'
act.name = "idle"

bpy.ops.object.mode_set(mode='OBJECT')
bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("POSE_IDLE_DONE")

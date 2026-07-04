"""Build the Weaver penthouse — a $1M night-penthouse scene — fully procedurally.

Run headless:  blender -b --factory-startup -P build_penthouse.py

Outputs weaver_apartment.glb next to this script (GLB, meshes+lights, no anims),
sized to the exact bounds embodiment.html's runtime atmosphere layer assumes:
  three.js x ∈ [-4.6, 4.6], z ∈ [-3.5 window wall … +3.2 back], ceiling y ≈ 2.9.
Blender is Z-up and glTF export maps X→X, Z→Y, +Y→-Z, so everything below is
placed with  y_blender = -z_threejs.

Composition targets the page's FIXED camera (0.15, 1.38, 2.75) → (0, 1.05, 0),
38° vfov: sofa left, marble island + pendants right, fire ribbon + skyline
center. The avatar stands at the origin — a 0.75 m radius there stays clear.
"""
import bpy
import numpy as np
import math
import os

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "weaver_apartment.glb")
rng = np.random.default_rng(7)

bpy.ops.wm.read_factory_settings(use_empty=True)
scene = bpy.context.scene
col = bpy.data.collections.new("Penthouse")
scene.collection.children.link(col)


# ───────────────────────── helpers ─────────────────────────
def link(ob):
    col.objects.link(ob)
    return ob


def mesh_ob(name, mesh):
    return link(bpy.data.objects.new(name, mesh))


def box(name, sx, sy, sz, x, y, z, mat, bevel=0.0):
    """Axis-aligned box, size (sx,sy,sz), CENTER at (x,y,z)."""
    m = bpy.data.meshes.new(name)
    ob = mesh_ob(name, m)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cube(bm, size=1.0)
    bm.to_mesh(m)
    bm.free()
    ob.scale = (sx, sy, sz)
    ob.location = (x, y, z)
    m.materials.append(mat)
    if bevel > 0:
        bv = ob.modifiers.new("bev", "BEVEL")
        bv.width = bevel
        bv.segments = 3
        bv.limit_method = "ANGLE"
    return ob


def cyl(name, r, depth, x, y, z, mat, verts=48, smooth=True):
    m = bpy.data.meshes.new(name)
    ob = mesh_ob(name, m)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_cone(bm, cap_ends=True, segments=verts, radius1=r, radius2=r, depth=depth)
    bm.to_mesh(m)
    bm.free()
    ob.location = (x, y, z)
    m.materials.append(mat)
    if smooth:
        for p in m.polygons:
            if abs(p.normal.z) < 0.9:
                p.use_smooth = True
    return ob


def sphere(name, r, x, y, z, mat, seg=32):
    m = bpy.data.meshes.new(name)
    ob = mesh_ob(name, m)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_uvsphere(bm, u_segments=seg, v_segments=seg // 2, radius=r)
    bm.to_mesh(m)
    bm.free()
    ob.location = (x, y, z)
    m.materials.append(mat)
    for p in m.polygons:
        p.use_smooth = True
    return ob


def plane(name, sx, sy, x, y, z, mat, rot=(0, 0, 0)):
    m = bpy.data.meshes.new(name)
    ob = mesh_ob(name, m)
    import bmesh
    bm = bmesh.new()
    bmesh.ops.create_grid(bm, x_segments=1, y_segments=1, size=0.5)
    bm.to_mesh(m)
    bm.free()
    # create_grid has no UVs by default in bmesh path — add simple 0..1 UVs
    uv = m.uv_layers.new(name="UVMap")
    for li, l in enumerate(m.loops):
        co = m.vertices[l.vertex_index].co
        uv.data[li].uv = (0.5 + co.x, 0.5 + co.y)   # grid is 1x1 centered — exact corner UVs
    ob.scale = (sx, sy, 1)
    ob.location = (x, y, z)
    ob.rotation_euler = rot
    m.materials.append(mat)
    return ob


def point_light(name, x, y, z, color, watts):
    li = bpy.data.lights.new(name, "POINT")
    li.color = color
    li.energy = watts
    li.shadow_soft_size = 0.3
    ob = bpy.data.objects.new(name, li)
    ob.location = (x, y, z)
    return link(ob)


# ───────────────────────── materials ─────────────────────────
def pbr(name, color, rough=0.5, metal=0.0, emit=None, emit_str=0.0, alpha=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    b = m.node_tree.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_str
    if alpha < 1.0:
        b.inputs["Alpha"].default_value = alpha
        m.blend_method = "BLEND"
    return m


def image_from_array(name, arr):
    """arr: HxWx3 float 0..1 → packed Blender image."""
    h, w, _ = arr.shape
    img = bpy.data.images.new(name, width=w, height=h, alpha=False)
    rgba = np.ones((h, w, 4), dtype=np.float32)
    rgba[:, :, :3] = arr
    img.pixels.foreach_set(rgba.ravel())
    img.pack()
    return img


def tex_material(name, img, rough=0.5, metal=0.0, emissive=False, emit_str=1.0, uv_scale=1.0):
    m = bpy.data.materials.new(name)
    m.use_nodes = True
    nt = m.node_tree
    b = nt.nodes["Principled BSDF"]
    t = nt.nodes.new("ShaderNodeTexImage")
    t.image = img
    if uv_scale != 1.0:
        mp = nt.nodes.new("ShaderNodeMapping")
        uvn = nt.nodes.new("ShaderNodeTexCoord")
        mp.inputs["Scale"].default_value = (uv_scale, uv_scale, 1)
        nt.links.new(uvn.outputs["UV"], mp.inputs["Vector"])
        nt.links.new(mp.outputs["Vector"], t.inputs["Vector"])
    if emissive:
        b.inputs["Base Color"].default_value = (0, 0, 0, 1)
        nt.links.new(t.outputs["Color"], b.inputs["Emission Color"])
        b.inputs["Emission Strength"].default_value = emit_str
        b.inputs["Roughness"].default_value = 1.0
    else:
        nt.links.new(t.outputs["Color"], b.inputs["Base Color"])
        b.inputs["Roughness"].default_value = rough
        b.inputs["Metallic"].default_value = metal
    return m


def fbm(w, h, octaves=5, seed=0):
    r = np.random.default_rng(seed)
    out = np.zeros((h, w))
    for o in range(octaves):
        s = 2 ** o
        small = r.random((h // (2 ** (octaves - 1 - o)) + 2, w // (2 ** (octaves - 1 - o)) + 2))
        yy = np.linspace(0, small.shape[0] - 1.001, h)
        xx = np.linspace(0, small.shape[1] - 1.001, w)
        yi, xi = np.floor(yy).astype(int), np.floor(xx).astype(int)
        yf, xf = (yy - yi)[:, None], (xx - xi)[None, :]
        a = small[yi][:, xi]
        bq = small[yi][:, xi + 1]
        c = small[yi + 1][:, xi]
        d = small[yi + 1][:, xi + 1]
        out += ((a * (1 - xf) + bq * xf) * (1 - yf) + (c * (1 - xf) + d * xf) * yf) / (2 ** (octaves - 1 - o))
    return (out - out.min()) / (out.max() - out.min() + 1e-9)


def make_marble(size=1024):
    base = np.array([0.492, 0.462, 0.424])
    n1 = fbm(size, size, 6, seed=11)
    warp = fbm(size, size, 5, seed=23)
    veins = np.abs(np.sin((n1 * 6.0 + warp * 4.0) * math.pi))
    veins = np.clip(1.0 - veins, 0, 1) ** 4.5
    gold = np.clip(1.0 - np.abs(np.sin((warp * 9.0 + n1 * 2.0) * math.pi)), 0, 1) ** 14
    img = np.ones((size, size, 3)) * base
    img -= veins[..., None] * np.array([0.30, 0.29, 0.27])          # grey veining
    img += gold[..., None] * np.array([0.06, 0.015, -0.05])         # faint gold thread
    img -= (fbm(size, size, 4, seed=5)[..., None] - 0.5) * 0.045    # tonal drift
    return np.clip(img, 0, 1)


def make_walnut(size=512):
    y = np.linspace(0, 1, size)[:, None] * np.ones((1, size))
    grain = fbm(size, size, 5, seed=31)
    stripes = 0.5 + 0.5 * np.sin((y * 22 + grain * 3.5) * math.pi)
    dark = np.array([0.145, 0.086, 0.052])
    light = np.array([0.265, 0.166, 0.098])
    img = dark + (light - dark) * (stripes ** 1.4)[..., None]
    img *= 0.9 + fbm(size, size, 3, seed=41)[..., None] * 0.2
    return np.clip(img, 0, 1)


def make_skyline(w=2048, h=768):
    """Night city panorama: layered silhouettes, lit windows, haze, moon."""
    img = np.zeros((h, w, 3))
    yy = np.linspace(1, 0, h)[:, None]                    # 1 at bottom row (image row 0 = bottom in Blender)
    # sky gradient: deep blue-black up, warm haze at horizon
    img[:, :, 0] = 0.010 + 0.045 * (yy ** 2.4)
    img[:, :, 1] = 0.013 + 0.036 * (yy ** 2.6)
    img[:, :, 2] = 0.030 + 0.050 * (yy ** 1.8)
    img += (yy ** 6)[..., None] * np.array([0.055, 0.032, 0.012])   # sodium glow at horizon
    # moon + glow, upper right
    mx, my, mr = int(w * 0.80), int(h * 0.82), h * 0.045
    X, Y = np.meshgrid(np.arange(w), np.arange(h))
    d = np.sqrt((X - mx) ** 2 + (Y - my) ** 2)
    img += np.clip(1 - d / (mr * 6), 0, 1)[..., None] ** 3 * np.array([0.10, 0.11, 0.13])
    img = np.where((d < mr)[..., None], np.array([0.92, 0.93, 0.90]), img)
    r = np.random.default_rng(77)
    # two silhouette layers, far dimmer than near
    for layer, (top_lo, top_hi, tone, lit_p, bright) in enumerate([
        (0.30, 0.62, np.array([0.014, 0.017, 0.028]), 0.13, 0.55),
        (0.14, 0.50, np.array([0.008, 0.010, 0.018]), 0.22, 1.00),
    ]):
        x = 0
        while x < w:
            bw = int(r.uniform(0.018, 0.05) * w)
            bh = int(r.uniform(top_lo, top_hi) * h)
            img[:bh, x:x + bw] = tone
            # window grid
            win_w, win_h, gap = max(3, bw // 14), 4, 3
            for wy in range(6, bh - 6, win_h + gap):
                for wx in range(x + 4, x + bw - 4, win_w + gap):
                    if r.random() < lit_p:
                        c = r.choice(3)
                        colr = [np.array([1.0, 0.82, 0.48]), np.array([0.62, 0.75, 1.0]),
                                np.array([0.95, 0.95, 0.9])][c]
                        img[wy:wy + win_h - 1, wx:wx + win_w - 1] = colr * bright * r.uniform(0.5, 1.0)
            if layer == 1 and r.random() < 0.25 and bh > h * 0.4:   # red aviation beacon
                img[bh - 2:bh, x + bw // 2 - 1:x + bw // 2 + 1] = np.array([1.0, 0.15, 0.1])
            x += bw + int(r.uniform(0.001, 0.012) * w)
    return np.clip(img, 0, 1)


marble_img = image_from_array("tx_marble", make_marble())
walnut_img = image_from_array("tx_walnut", make_walnut())
skyline_img = image_from_array("tx_skyline", make_skyline())

M = {
    "floor":    tex_material("marble_floor", marble_img, rough=0.32, uv_scale=2.5),
    "island":   tex_material("marble_island", marble_img, rough=0.22, uv_scale=1.2),
    "walnut":   tex_material("walnut", walnut_img, rough=0.45, uv_scale=1.0),
    "skyline":  tex_material("skyline", skyline_img, emissive=True, emit_str=1.9),
    "wall":     pbr("wall_greige", (0.318, 0.296, 0.268), rough=0.85),
    "ceiling":  pbr("ceiling_char", (0.062, 0.060, 0.058), rough=0.9),
    "mullion":  pbr("bronze_mullion", (0.055, 0.045, 0.038), rough=0.4, metal=0.9),
    "glass":    pbr("glazing", (0.72, 0.84, 1.0), rough=0.03, alpha=0.04),
    "boucle":   pbr("boucle_ivory", (0.545, 0.512, 0.458), rough=0.95),
    "cushion2": pbr("boucle_sand", (0.412, 0.368, 0.306), rough=0.95),
    "leather":  pbr("leather_tan", (0.262, 0.132, 0.066), rough=0.5),
    "brass":    pbr("brass", (0.788, 0.640, 0.336), rough=0.32, metal=1.0),
    "black":    pbr("matte_black", (0.028, 0.028, 0.030), rough=0.6),
    "granite":  pbr("black_granite", (0.035, 0.034, 0.036), rough=0.18),
    "rug":      pbr("rug_ivory", (0.435, 0.398, 0.340), rough=1.0),
    "rugring":  pbr("rug_ring", (0.148, 0.118, 0.088), rough=1.0),
    "ceramic":  pbr("ceramic_pot", (0.208, 0.196, 0.180), rough=0.7),
    "leaf":     pbr("leaf", (0.083, 0.190, 0.081), rough=0.7),
    "trunk":    pbr("trunk", (0.196, 0.135, 0.083), rough=0.9),
    "fire":     pbr("fire_ribbon", (0.05, 0.02, 0.0), rough=1.0, emit=(1.0, 0.42, 0.10), emit_str=6.0),
    "lampglow": pbr("lamp_globe", (1, 1, 1), rough=1.0, emit=(1.0, 0.85, 0.62), emit_str=4.0),
    "spot":     pbr("can_light", (0, 0, 0), rough=1.0, emit=(1.0, 0.88, 0.70), emit_str=3.0),
    "shelf_glow": pbr("shelf_glow", (0, 0, 0), rough=1.0, emit=(1.0, 0.78, 0.45), emit_str=2.2),
    "bottle":   pbr("bottle", (0.14, 0.28, 0.16), rough=0.1, alpha=0.5),
    "duvet":    pbr("duvet", (0.52, 0.50, 0.47), rough=0.95),
    "throw":    pbr("throw_rust", (0.52, 0.24, 0.12), rough=0.95),
    "art1":     pbr("art_navy", (0.055, 0.078, 0.155), rough=0.8),
    "art2":     pbr("art_gold", (0.72, 0.55, 0.25), rough=0.5, metal=0.6),
    "deck":     pbr("deck_wood", (0.165, 0.120, 0.080), rough=0.8),
}

# ─────────────── shell: floor / walls / ceiling / glazing ───────────────
# Blender y = -three.js z. Window wall three z=-3.5 → blender y=+3.5.
plane("floor", 9.6, 7.2, 0, 0.15, 0.0, M["floor"])                       # three z: -3.75..+3.45
plane("ceiling", 9.6, 7.2, 0, 0.15, 2.94, M["ceiling"], rot=(math.pi, 0, 0))
# perimeter cove fascia (drops below ceiling, leaves lit channel near walls)
box("fascia_win", 9.6, 0.5, 0.24, 0, 3.20, 2.82, M["ceiling"])
box("fascia_back", 9.6, 0.5, 0.24, 0, -2.95, 2.82, M["ceiling"])
box("fascia_L", 0.5, 7.2, 0.24, -4.45, 0.15, 2.82, M["ceiling"])
box("fascia_R", 0.5, 7.2, 0.24, 4.45, 0.15, 2.82, M["ceiling"])
box("wall_L", 0.1, 7.2, 3.0, -4.75, 0.15, 1.5, M["wall"])
box("wall_R", 0.1, 7.2, 3.0, 4.75, 0.15, 1.5, M["wall"])
box("wall_back", 9.6, 0.1, 3.0, 0, -3.30, 1.5, M["wall"])

# window wall: glass + bronze mullions
plane("glazing", 9.4, 2.9, 0, 3.52, 1.45, M["glass"], rot=(math.pi / 2, 0, 0))
box("rail_top", 9.5, 0.09, 0.10, 0, 3.53, 2.87, M["mullion"])
box("rail_bot", 9.5, 0.09, 0.07, 0, 3.53, 0.035, M["mullion"])
for i in range(9):
    x = -4.6 + i * 1.15
    box(f"mullion_{i}", 0.055, 0.075, 2.9, x, 3.53, 1.45, M["mullion"])

# outside: balcony deck + glass rail + skyline
box("balcony_deck", 9.6, 1.15, 0.05, 0, 4.15, -0.03, M["deck"])
plane("balcony_rail", 9.4, 1.05, 0, 4.70, 0.55, M["glass"], rot=(math.pi / 2, 0, 0))
box("rail_cap", 9.4, 0.05, 0.04, 0, 4.70, 1.09, M["mullion"])
plane("skyline", 26, 9.0, 0, 6.4, 3.1, M["skyline"], rot=(math.pi / 2, 0, 0))

# ───────────────────────── lounge (left of her) ─────────────────────────
# rug (she stands on its edge at origin)
cyl("rug", 2.15, 0.012, -0.7, 0.95, 0.006, M["rug"], verts=64)
cyl("rug_ring", 2.15, 0.013, -0.7, 0.95, 0.0065, M["rugring"], verts=64).scale = (0.999, 0.999, 1)
cyl("rug_inner", 1.85, 0.014, -0.7, 0.95, 0.007, M["rug"], verts=64)

# sofa: platform + seat + back + arm cushions (faces the camera, back to window)
sofa_y = 1.9        # three z = -1.9
box("sofa_base", 2.9, 1.05, 0.16, -1.45, sofa_y, 0.10, M["walnut"], bevel=0.015)
box("sofa_seat_L", 1.34, 0.95, 0.17, -2.14, sofa_y, 0.27, M["boucle"], bevel=0.05)
box("sofa_seat_R", 1.34, 0.95, 0.17, -0.76, sofa_y, 0.27, M["boucle"], bevel=0.05)
box("sofa_back_L", 1.34, 0.24, 0.42, -2.14, sofa_y + 0.42, 0.58, M["boucle"], bevel=0.06)
box("sofa_back_R", 1.34, 0.24, 0.42, -0.76, sofa_y + 0.42, 0.58, M["boucle"], bevel=0.06)
box("sofa_arm_L", 0.22, 1.05, 0.34, -2.90, sofa_y, 0.44, M["boucle"], bevel=0.05)
box("sofa_arm_R", 0.22, 1.05, 0.34, 0.0, sofa_y, 0.44, M["boucle"], bevel=0.05)
box("pillow_1", 0.42, 0.14, 0.30, -2.35, sofa_y + 0.36, 0.52, M["cushion2"], bevel=0.05)
box("pillow_2", 0.42, 0.14, 0.30, -0.55, sofa_y + 0.36, 0.52, M["cushion2"], bevel=0.05)
box("pillow_3", 0.38, 0.13, 0.27, -1.35, sofa_y + 0.38, 0.50, M["throw"], bevel=0.05)
box("sofa_seam", 0.03, 0.95, 0.16, -1.45, sofa_y, 0.27, M["black"])

# marble coffee table drum + brass tray
cyl("coffee_table", 0.52, 0.34, -1.05, 1.0, 0.17, M["granite"], verts=64)
cyl("coffee_inlay", 0.525, 0.015, -1.05, 1.0, 0.345, M["brass"], verts=64)
cyl("tray", 0.20, 0.02, -1.15, 0.92, 0.35, M["brass"], verts=48)
box("book_1", 0.26, 0.19, 0.025, -0.85, 1.08, 0.353, M["art1"])
box("book_2", 0.23, 0.17, 0.02, -0.86, 1.07, 0.376, M["art2"])

# tan leather accent chair, angled toward sofa
chair = box("chair_seat", 0.62, 0.60, 0.14, 1.45, 2.15, 0.30, M["leather"], bevel=0.04)
chair_back = box("chair_back", 0.62, 0.15, 0.45, 1.45, 2.42, 0.60, M["leather"], bevel=0.05)
for ob in (chair, chair_back):
    ob.rotation_euler = (0, 0, math.radians(-24))
box("chair_legs", 0.5, 0.5, 0.16, 1.45, 2.2, 0.08, M["brass"]).rotation_euler = (0, 0, math.radians(-24))

# arc floor lamp: base, stem (arched via 3 segments), emissive globe over sofa
cyl("lamp_base", 0.17, 0.03, -2.55, 2.6, 0.015, M["black"], verts=32)
s1 = cyl("lamp_stem1", 0.016, 1.9, -2.55, 2.6, 0.97, M["brass"], verts=16)
s2 = cyl("lamp_stem2", 0.015, 1.1, -2.30, 2.48, 2.05, M["brass"], verts=16)
s2.rotation_euler = (0.32, 0.55, 0)
sphere("lamp_globe", 0.13, -1.95, 2.28, 2.34, M["lampglow"], seg=24)

# fire ribbon under the window, right of the sofa
box("fire_plinth", 1.45, 0.34, 0.42, 1.35, 3.28, 0.21, M["granite"], bevel=0.012)
box("fire_ribbon", 1.15, 0.05, 0.10, 1.35, 3.20, 0.34, M["fire"])
box("fire_ledge", 1.45, 0.36, 0.02, 1.35, 3.28, 0.43, M["granite"])

# console under window (left) + sculpture + art lean
box("console", 1.3, 0.32, 0.04, -2.6, 3.22, 0.72, M["walnut"], bevel=0.01)
box("console_leg_L", 0.03, 0.28, 0.70, -3.18, 3.22, 0.35, M["black"])
box("console_leg_R", 0.03, 0.28, 0.70, -2.02, 3.22, 0.35, M["black"])
sphere("sculpt_orb", 0.09, -2.85, 3.2, 0.84, M["brass"], seg=24)
box("art_lean", 0.72, 0.03, 0.95, -2.35, 3.30, 1.24, M["art1"], bevel=0.005)
box("art_lean_frame", 0.78, 0.02, 1.01, -2.35, 3.32, 1.24, M["brass"])

# ─────────────────── kitchen island + pendants (right) ───────────────────
isl_x, isl_y = 2.45, 2.3          # three z = -2.3
box("island_body", 1.5, 0.78, 0.86, isl_x, isl_y, 0.44, M["walnut"], bevel=0.01)
box("island_top", 1.66, 0.92, 0.05, isl_x, isl_y, 0.905, M["island"], bevel=0.008)
box("island_fall_L", 0.05, 0.92, 0.90, isl_x - 0.805, isl_y, 0.45, M["island"])
box("island_fall_R", 0.05, 0.92, 0.90, isl_x + 0.805, isl_y, 0.45, M["island"])
box("island_kick", 1.44, 0.72, 0.08, isl_x, isl_y, 0.045, M["black"])
# stools camera-side
for i, sx in enumerate((isl_x - 0.4, isl_x + 0.4)):
    cyl(f"stool_seat_{i}", 0.19, 0.06, sx, isl_y - 0.75, 0.66, M["leather"], verts=32)
    cyl(f"stool_leg_{i}", 0.025, 0.62, sx, isl_y - 0.75, 0.33, M["brass"], verts=16)
    cyl(f"stool_foot_{i}", 0.14, 0.02, sx, isl_y - 0.75, 0.02, M["brass"], verts=24)
# three brass pendants with emissive discs + cables
for i, px in enumerate((isl_x - 0.5, isl_x, isl_x + 0.5)):
    cyl(f"pend_cable_{i}", 0.006, 0.75, px, isl_y, 2.55, M["black"], verts=8)
    cyl(f"pend_shade_{i}", 0.09, 0.16, px, isl_y, 2.10, M["brass"], verts=32)
    cyl(f"pend_glow_{i}", 0.075, 0.012, px, isl_y, 2.015, M["lampglow"], verts=24)
# decor: fruit bowl + wine
cyl("bowl", 0.14, 0.05, isl_x - 0.35, isl_y + 0.1, 0.955, M["granite"], verts=32)
cyl("wine_bottle", 0.045, 0.30, isl_x + 0.45, isl_y + 0.18, 1.08, M["bottle"], verts=16)

# bar shelving unit on the right wall (runtime wash at x≈4.23 grazes it)
box("bar_body", 0.32, 2.2, 2.3, 4.55, 0.9, 1.15, M["walnut"])
for i in range(3):
    z = 0.75 + i * 0.55
    box(f"bar_shelf_{i}", 0.30, 2.0, 0.025, 4.54, 0.9, z, M["black"])
    box(f"bar_glow_{i}", 0.02, 1.9, 0.012, 4.42, 0.9, z + 0.012, M["shelf_glow"])
    for j in range(4):
        cyl(f"bottle_{i}_{j}", 0.032, 0.26, 4.52, 0.15 + j * 0.42, z + 0.16, M["bottle"], verts=12)

# ───────────────── bedroom nook (behind camera, completeness) ─────────────────
bed_x, bed_y = -3.05, -2.15       # three z = +2.15
box("bed_platform", 2.1, 1.9, 0.18, bed_x, bed_y, 0.09, M["walnut"], bevel=0.01)
box("bed_mattress", 1.9, 1.7, 0.22, bed_x, bed_y, 0.30, M["duvet"], bevel=0.06)
box("bed_duvet", 1.92, 1.2, 0.10, bed_x, bed_y - 0.25, 0.435, M["duvet"], bevel=0.05)
box("bed_throw", 1.92, 0.45, 0.06, bed_x, bed_y - 0.60, 0.46, M["throw"], bevel=0.03)
box("headboard", 2.2, 0.08, 1.1, bed_x, bed_y + 0.92, 0.75, M["boucle"], bevel=0.03)
box("pillow_b1", 0.62, 0.16, 0.24, bed_x - 0.48, bed_y + 0.62, 0.52, M["duvet"], bevel=0.06)
box("pillow_b2", 0.62, 0.16, 0.24, bed_x + 0.48, bed_y + 0.62, 0.52, M["duvet"], bevel=0.06)
for i, nx in enumerate((bed_x - 1.25, bed_x + 1.25)):
    box(f"nightstand_{i}", 0.42, 0.38, 0.42, nx, bed_y + 0.6, 0.21, M["walnut"], bevel=0.01)
    sphere(f"night_lamp_{i}", 0.07, nx, bed_y + 0.6, 0.50, M["lampglow"], seg=16)
# large abstract art on back wall
box("art_back", 1.7, 0.03, 1.15, 0.8, -3.26, 1.55, M["art1"])
box("art_back_frame", 1.78, 0.02, 1.23, 0.8, -3.28, 1.55, M["brass"])
box("art_back_accent", 0.6, 0.035, 1.15, 0.45, -3.255, 1.55, M["art2"])

# ───────────────────────── plants ─────────────────────────
def plant(name, x, y, scale=1.0, kind="fiddle"):
    cyl(f"{name}_pot", 0.20 * scale, 0.34 * scale, x, y, 0.17 * scale, M["ceramic"], verts=32)
    cyl(f"{name}_trunk", 0.025 * scale, 0.9 * scale, x, y, 0.75 * scale, M["trunk"], verts=12)
    r = np.random.default_rng(hash(name) % 2**31)
    n = 9 if kind == "fiddle" else 7
    for i in range(n):
        a = r.uniform(0, 2 * math.pi)
        rad = r.uniform(0.08, 0.30) * scale
        h = (1.05 + r.uniform(0, 0.55)) * scale
        s = sphere(f"{name}_leaf{i}", r.uniform(0.13, 0.22) * scale,
                   x + math.cos(a) * rad, y + math.sin(a) * rad, h, M["leaf"], seg=12)
        s.scale = (1, 1, r.uniform(0.6, 0.85))

plant("plant_L", -3.35, 2.9, 1.15)
plant("plant_R", 3.65, 2.95, 1.0)
plant("plant_bed", -4.2, -0.5, 0.8)

# ───────────────────── ceiling can lights (emissive discs) ─────────────────────
for i, (cx, cy) in enumerate([(-1.4, 1.9), (-1.4, 0.0), (2.45, 2.3), (2.45, 0.2),
                              (0.4, 3.0), (-3.05, -2.15), (0.8, -2.6)]):
    cyl(f"can_{i}", 0.055, 0.012, cx, cy, 2.935, M["spot"], verts=20)

# ───────────────────────── lights (page clamps intensity ≤60) ─────────────────────────
point_light("L_lounge", -1.4, 1.6, 2.5, (1.0, 0.83, 0.62), 0.9)
point_light("L_island", 2.45, 2.3, 2.3, (1.0, 0.85, 0.65), 0.8)
point_light("L_fire", 1.45, 3.0, 0.6, (1.0, 0.48, 0.15), 0.55)
point_light("L_window_cool", 0.0, 3.2, 1.8, (0.62, 0.74, 1.0), 0.25)
point_light("L_bed", -3.05, -2.0, 1.8, (1.0, 0.80, 0.60), 0.35)

# ───────────────────────── export ─────────────────────────
kwargs = dict(filepath=OUT, export_format="GLB", export_apply=True, export_animations=False,
              export_yup=True)
props = bpy.ops.export_scene.gltf.get_rna_type().properties.keys()
if "export_lights" in props:
    kwargs["export_lights"] = True
if "export_image_format" in props:
    kwargs["export_image_format"] = "AUTO"
bpy.ops.export_scene.gltf(**kwargs)
print(f"PENTHOUSE_EXPORTED {OUT} ({os.path.getsize(OUT)/1e6:.1f} MB, "
      f"{len([o for o in col.objects if o.type=='MESH'])} meshes)")

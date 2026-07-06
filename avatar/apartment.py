"""High-end modern penthouse around the avatar (GTA-online vibe) + African figurines."""
import bpy, bmesh, math, random
import numpy as np
from mathutils import Vector, Euler

random.seed(7)
S = bpy.context.scene
col = bpy.data.collections.new("Apartment")
S.collection.children.link(col)

def mat(name, color, rough=0.5, metal=0.0, emit=None, emit_str=1.0):
    m = bpy.data.materials.new(name); m.use_nodes = True
    nt = m.node_tree; b = nt.nodes["Principled BSDF"]
    b.inputs["Base Color"].default_value = (*color, 1)
    b.inputs["Roughness"].default_value = rough
    b.inputs["Metallic"].default_value = metal
    if emit:
        b.inputs["Emission Color"].default_value = (*emit, 1)
        b.inputs["Emission Strength"].default_value = emit_str
    return m

M_floor  = mat("MarbleFloor", (0.82, 0.80, 0.78), 0.08)
M_wall   = mat("WallWarm", (0.28, 0.26, 0.24), 0.8)
M_accent = mat("WallDark", (0.10, 0.09, 0.09), 0.6)
M_gold   = mat("GoldTrim", (1.0, 0.72, 0.25), 0.2, 1.0)
M_sofa   = mat("SofaCream", (0.85, 0.80, 0.72), 0.75)
M_wood   = mat("DarkWood", (0.09, 0.055, 0.035), 0.35)
M_ebony  = mat("Ebony", (0.05, 0.032, 0.022), 0.28)
M_glass  = mat("Glass", (0.9, 0.95, 1.0), 0.03); M_glass.node_tree.nodes["Principled BSDF"].inputs["Alpha"].default_value = 0.12; M_glass.blend_method='BLEND'
M_rug    = mat("Rug", (0.16, 0.13, 0.11), 0.95)
M_screen = mat("TVScreen", (0.02,0.02,0.02), 0.3, emit=(0.10, 0.15, 0.3), emit_str=2.0)
M_strip  = mat("LightStrip", (1,1,1), 0.5, emit=(1.0, 0.85, 0.6), emit_str=8.0)

def box(name, size, loc, m, rot=(0,0,0), bevel=0.0):
    bpy.ops.mesh.primitive_cube_add(size=1, location=loc, rotation=rot)
    ob = bpy.context.active_object; ob.name = name
    ob.scale = (size[0]/2, size[1]/2, size[2]/2)
    bpy.ops.object.transform_apply(scale=True)
    if bevel > 0:
        bv = ob.modifiers.new("B", "BEVEL"); bv.width = bevel; bv.segments = 3
    ob.data.materials.append(m)
    for c in ob.users_collection: c.objects.unlink(ob)
    col.objects.link(ob)
    return ob

# ── shell: floor 9x7, walls, window wall at y=-3 ──
box("floor", (9, 7, 0.1), (0, 0, -0.05), M_floor)
box("ceiling", (9, 7, 0.1), (0, 0, 3.05), M_wall)
box("wall_back", (9, 0.1, 3), (0, 3.5, 1.5), M_wall)
box("wall_left", (0.1, 7, 3), (-4.5, 0, 1.5), M_accent)
box("wall_right", (0.1, 7, 3), (4.5, 0, 1.5), M_wall)
# window wall: glass panes + gold mullions
for i in range(5):
    x = -3.6 + i*1.8
    box(f"mullion{i}", (0.06, 0.06, 3), (x, -3.5, 1.5), M_gold)
box("mullion_top", (9, 0.06, 0.08), (0, -3.5, 2.96), M_gold)
box("mullion_bot", (9, 0.06, 0.12), (0, -3.5, 0.06), M_gold)
box("window_glass", (8.9, 0.02, 2.9), (0, -3.52, 1.5), M_glass)

# ── night-city backdrop texture (numpy) ──
W, H = 2048, 1024
img = np.zeros((H, W, 4), dtype=np.float32); img[..., 3] = 1
for y in range(H):
    t = y / H
    img[y, :, 0] = 0.008 + 0.02*t; img[y, :, 1] = 0.01 + 0.03*t; img[y, :, 2] = 0.03 + 0.07*t
x = 30
while x < W - 40:                       # buildings
    bw = random.randint(40, 130); bh = random.randint(int(H*0.25), int(H*0.75))
    for wx in range(x+6, x+bw-6, 14):
        for wy in range(40, bh, 22):
            if random.random() < 0.45:
                c = random.choice([(1.0,0.85,0.5),(0.6,0.8,1.0),(1.0,0.95,0.8)])
                br = random.uniform(0.5, 2.0)
                img[wy:wy+8, wx:wx+8, :3] = np.array(c)*br
    x += bw + random.randint(6, 30)
city = bpy.data.images.new("city_night", W, H, alpha=True)
city.pixels = img.ravel().tolist()
city.filepath_raw = "/media/ydn/SYPHER_CORE2/weaver v3/avatar/textures/city_night.png"
city.file_format = 'PNG'; city.save()
M_city = bpy.data.materials.new("CityNight"); M_city.use_nodes = True
nt = M_city.node_tree; nt.nodes.clear()
out = nt.nodes.new("ShaderNodeOutputMaterial"); em = nt.nodes.new("ShaderNodeEmission")
t = nt.nodes.new("ShaderNodeTexImage"); t.image = city
em.inputs["Strength"].default_value = 1.6
nt.links.new(t.outputs["Color"], em.inputs["Color"]); nt.links.new(em.outputs[0], out.inputs[0])
back = box("city_backdrop", (22, 0.05, 9), (0, -8.5, 3.2), M_city)

# ── furniture ──
box("sofa_seat", (3.0, 1.05, 0.42), (0.6, 2.35, 0.21), M_sofa, bevel=0.05)
box("sofa_back", (3.0, 0.25, 0.55), (0.6, 2.82, 0.75), M_sofa, bevel=0.05)
box("sofa_armL", (0.28, 1.05, 0.62), (-0.95, 2.35, 0.31), M_sofa, bevel=0.05)
box("sofa_armR", (0.28, 1.05, 0.62), (2.15, 2.35, 0.31), M_sofa, bevel=0.05)
box("sofa_L_seat", (1.0, 1.9, 0.42), (-1.9, 1.6, 0.21), M_sofa, bevel=0.05)
box("sofa_L_back", (0.25, 1.9, 0.55), (-2.32, 1.6, 0.75), M_sofa, bevel=0.05)
box("rug", (3.6, 2.6, 0.02), (0.3, 0.9, 0.012), M_rug, bevel=0.01)
box("ctable_top", (1.5, 0.8, 0.04), (0.3, 0.9, 0.42), M_glass)
for dx, dy in ((-0.65,-0.32),(0.65,-0.32),(-0.65,0.32),(0.65,0.32)):
    box(f"ctleg{dx}{dy}", (0.05, 0.05, 0.4), (0.3+dx, 0.9+dy, 0.2), M_gold)
box("tv_console", (2.6, 0.45, 0.45), (0, 3.32, 0.225), M_wood, bevel=0.02)
box("tv", (2.2, 0.06, 1.15), (0, 3.42, 1.35), M_screen)
box("art1", (0.9, 0.05, 1.2), (-4.42, -1.2, 1.7), mat("Art1", (0.7,0.3,0.1), 0.6, emit=(0.9,0.4,0.12), emit_str=0.5), rot=(0,0,0))
box("art2", (0.9, 0.05, 1.2), (-4.42, 0.4, 1.7), mat("Art2", (0.1,0.3,0.5), 0.6, emit=(0.15,0.35,0.6), emit_str=0.5))
box("shelf", (0.4, 1.6, 0.06), (4.25, -0.5, 1.3), M_wood)
box("pedestal1", (0.35, 0.35, 1.0), (3.9, -2.6, 0.5), M_floor, bevel=0.01)
box("pedestal2", (0.35, 0.35, 0.8), (-3.9, -2.4, 0.4), M_floor, bevel=0.01)
box("strip1", (6, 0.12, 0.02), (0, -1.0, 2.98), M_strip)
box("strip2", (6, 0.12, 0.02), (0, 1.6, 2.98), M_strip)

# ── African figurines (elongated carved style) ──
def figurine(name, loc, scale=1.0, rings=4):
    zs = loc[2]
    parts = []
    def add(ob): parts.append(ob); return ob
    bpy.ops.mesh.primitive_cone_add(radius1=0.09*scale, radius2=0.05*scale, depth=0.55*scale,
                                    location=(loc[0], loc[1], zs+0.28*scale))
    body = bpy.context.active_object; add(body)
    for i in range(rings):                                    # neck rings
        bpy.ops.mesh.primitive_torus_add(major_radius=0.045*scale, minor_radius=0.012*scale,
                                         location=(loc[0], loc[1], zs+(0.58+0.045*i)*scale))
        r = bpy.context.active_object; r.data.materials.append(M_gold); add(r)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=0.085*scale,
                                         location=(loc[0], loc[1], zs+(0.62+0.045*rings+0.06)*scale))
    head = bpy.context.active_object
    head.scale = (0.72, 0.8, 1.45); add(head)
    bpy.ops.mesh.primitive_cylinder_add(radius=0.12*scale, depth=0.05*scale,
                                        location=(loc[0], loc[1], zs+0.02*scale))
    base = bpy.context.active_object; add(base)
    for ob in parts:
        if not ob.data.materials: ob.data.materials.append(M_ebony)
        bpy.ops.object.shade_smooth()
        for c in ob.users_collection: c.objects.unlink(ob)
        col.objects.link(ob)
    return parts

figurine("fig1", (3.9, -2.6, 1.0), scale=1.0)
figurine("fig2", (-3.9, -2.4, 0.8), scale=0.8, rings=3)
figurine("fig3", (-3.4, 2.9, 0.0), scale=2.6, rings=5)     # big floor statue
figurine("fig4", (4.25, -0.3, 1.33), scale=0.45, rings=3)  # shelf piece

# ── lighting: warm interior + city glow ──
def light(name, type, loc, energy, color=(1,1,1), size=1.0):
    l = bpy.data.lights.new(name, type); l.energy = energy; l.color = color
    if type == 'AREA': l.size = size
    ob = bpy.data.objects.new(name, l); ob.location = loc
    col.objects.link(ob)
    return ob
light("key_warm", 'AREA', (1.5, -0.5, 2.85), 220, (1.0, 0.85, 0.65), 2.5)
light("fill_cool", 'AREA', (-2.5, -2.8, 2.2), 90, (0.6, 0.7, 1.0), 3.0)
la = light("lamp_glow", 'POINT', (2.6, 2.9, 1.5), 60, (1.0, 0.7, 0.4))
box("lamp_pole", (0.04, 0.04, 1.5), (2.6, 2.9, 0.75), M_gold)
bpy.ops.mesh.primitive_uv_sphere_add(radius=0.14, location=(2.6, 2.9, 1.55))
sh = bpy.context.active_object; sh.data.materials.append(mat("LampShade",(1,0.8,0.55),0.5,emit=(1.0,0.75,0.45),emit_str=6.0))
for c in sh.users_collection: c.objects.unlink(sh)
col.objects.link(sh)
S.world.node_tree.nodes["Background"].inputs[0].default_value = (0.006, 0.007, 0.012, 1)

bpy.ops.wm.save_as_mainfile(filepath=bpy.data.filepath)
print("APARTMENT_DONE")

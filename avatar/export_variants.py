import bpy
OUTFITS = {
    "monokini": ["monokini", "belly_chain", "strap_chain"],
    "dress":    ["outfit_dress"],
    "casual":   ["outfit_crop_top", "outfit_jeans"],
    "sport":    ["outfit_sport_bra", "outfit_leggings"],
}
ALL = sorted({o for v in OUTFITS.values() for o in v})
for nm in ("PrevCam", "Key", "Fill"):
    ob = bpy.data.objects.get(nm)
    if ob: bpy.data.objects.remove(ob, do_unlink=True)
for outfit, show in OUTFITS.items():
    for nm in ALL:
        ob = bpy.data.objects.get(nm)
        if ob:
            hide = nm not in show
            ob.hide_render = hide
            ob.hide_viewport = hide     # use_visible follows viewport visibility
    bpy.ops.export_scene.gltf(
        filepath=f"/media/ydn/SYPHER_CORE2/weaver v3/avatar/weaver_avatar_{outfit}.glb",
        export_format='GLB', export_apply=True, export_animations=True,
        export_skins=True, export_yup=True, use_visible=True)
    print("EXPORTED", outfit)

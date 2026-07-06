import bpy
def collection_objects(collection):
    objects = set(collection.objects)
    for child in collection.children:
        objects.update(collection_objects(child))
    return objects

apt = collection_objects(bpy.data.collections["Apartment"])
for ob in bpy.data.objects:
    show = ob in apt and ob.type in ('MESH', 'LIGHT')
    ob.hide_viewport = not show
    ob.hide_render = not show
bpy.ops.export_scene.gltf(
    filepath="/media/ydn/SYPHER_CORE2/weaver v3/avatar/weaver_apartment.glb",
    export_format='GLB', export_apply=True, export_animations=False,
    export_skins=False, export_yup=True, use_visible=True, export_lights=True)
print("APT_GLB_DONE")

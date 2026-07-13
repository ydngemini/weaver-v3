#!/usr/bin/env python3
"""Build Weaver's high-fidelity skinned GLB and UV-aligned skin PBR maps.

The source identity, skeleton, animation accessors, texture coordinates, and
materials remain intact. Selected triangle meshes receive one deterministic
Phong-projected subdivision pass. New midpoint skin weights merge both endpoint
influences, keep the strongest four joints, and renormalize to the glTF skinning
contract.

This is an original asset-processing pipeline. It does not ingest or reproduce
third-party game geometry, textures, characters, or animation data.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import struct
import sys
from pathlib import Path
from typing import Iterable, Sequence

# avatar/inspect.py is a legacy Blender helper whose name would otherwise shadow
# Python's standard-library inspect module while NumPy imports.
SCRIPT_DIRECTORY = str(Path(__file__).resolve().parent)
sys.path = [entry for entry in sys.path if str(Path(entry or ".").resolve()) != SCRIPT_DIRECTORY]

import numpy as np
from PIL import Image, ImageFilter


ROOT = Path(__file__).resolve().parent
DEFAULT_SOURCE = ROOT / "weaver_avatar_dress.glb"
DEFAULT_OUTPUT = ROOT / "weaver_avatar_dress_hifi.glb"
SKIN_ALBEDO = ROOT / "textures" / "skin_dark.png"
SKIN_NORMAL = ROOT / "textures" / "skin_normal_hifi.png"
SKIN_ROUGHNESS = ROOT / "textures" / "skin_roughness_hifi.png"
SKIN_SPECULAR = ROOT / "textures" / "skin_specular_hifi.png"

JSON_CHUNK = 0x4E4F534A
BIN_CHUNK = 0x004E4942
GLB_MAGIC = 0x46546C67
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963

COMPONENT_FORMAT = {
    5120: "b",
    5121: "B",
    5122: "h",
    5123: "H",
    5125: "I",
    5126: "f",
}
COMPONENT_SIZE = {key: struct.calcsize("<" + value) for key, value in COMPONENT_FORMAT.items()}
TYPE_WIDTH = {"SCALAR": 1, "VEC2": 2, "VEC3": 3, "VEC4": 4, "MAT4": 16}

SUBDIVISION_PROFILES = {
    "weaver_base2-base": 0.58,
    "outfit_dress_mesh": 0.24,
    "weaver_base2-highpolyeyes": 0.82,
}


def read_glb(path: Path) -> tuple[dict, bytearray]:
    payload = path.read_bytes()
    magic, version, total_length = struct.unpack_from("<III", payload, 0)
    if magic != GLB_MAGIC or version != 2 or total_length != len(payload):
        raise ValueError(f"invalid GLB 2.0 container: {path}")
    offset = 12
    document = None
    binary = None
    while offset + 8 <= total_length:
        chunk_length, chunk_type = struct.unpack_from("<II", payload, offset)
        offset += 8
        chunk = payload[offset : offset + chunk_length]
        offset += chunk_length
        if chunk_type == JSON_CHUNK:
            document = json.loads(chunk.rstrip(b" \0"))
        elif chunk_type == BIN_CHUNK:
            binary = bytearray(chunk)
    if document is None or binary is None:
        raise ValueError("GLB requires one JSON chunk and one BIN chunk")
    if len(document.get("buffers", [])) != 1:
        raise ValueError("builder supports the single-buffer Blender GLB contract")
    return document, binary


def read_accessor(document: dict, binary: bytearray, accessor_index: int) -> list[tuple]:
    accessor = document["accessors"][accessor_index]
    if accessor.get("sparse"):
        raise ValueError("sparse accessors are not supported")
    component_type = accessor["componentType"]
    width = TYPE_WIDTH[accessor["type"]]
    component_size = COMPONENT_SIZE[component_type]
    packed_size = component_size * width
    view = document["bufferViews"][accessor["bufferView"]]
    stride = view.get("byteStride", packed_size)
    offset = view.get("byteOffset", 0) + accessor.get("byteOffset", 0)
    unpacker = struct.Struct("<" + COMPONENT_FORMAT[component_type] * width)
    return [unpacker.unpack_from(binary, offset + index * stride) for index in range(accessor["count"])]


def normalize(vector: Sequence[float]) -> tuple[float, float, float]:
    length = math.sqrt(sum(value * value for value in vector))
    if length <= 1e-12:
        return (0.0, 0.0, 1.0)
    return tuple(float(value / length) for value in vector)  # type: ignore[return-value]


def dot(a: Sequence[float], b: Sequence[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def add(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(x + y for x, y in zip(a, b))


def subtract(a: Sequence[float], b: Sequence[float]) -> tuple[float, ...]:
    return tuple(x - y for x, y in zip(a, b))


def scale(vector: Sequence[float], factor: float) -> tuple[float, ...]:
    return tuple(value * factor for value in vector)


def mix(a: Sequence[float], b: Sequence[float], factor: float = 0.5) -> tuple[float, ...]:
    return tuple(x + (y - x) * factor for x, y in zip(a, b))


def curved_midpoint(
    point_a: Sequence[float],
    point_b: Sequence[float],
    normal_a: Sequence[float],
    normal_b: Sequence[float],
    curvature: float,
) -> tuple[float, float, float]:
    linear = mix(point_a, point_b)
    normal_a = normalize(normal_a)
    normal_b = normalize(normal_b)
    agreement = dot(normal_a, normal_b)
    if agreement < 0.20 or curvature <= 0:
        return linear  # type: ignore[return-value]
    projected_a = subtract(linear, scale(normal_a, dot(subtract(linear, point_a), normal_a)))
    projected_b = subtract(linear, scale(normal_b, dot(subtract(linear, point_b), normal_b)))
    projected = mix(projected_a, projected_b)
    displacement = subtract(projected, linear)
    edge_length = math.sqrt(dot(subtract(point_b, point_a), subtract(point_b, point_a)))
    displacement_length = math.sqrt(dot(displacement, displacement))
    max_displacement = edge_length * 0.16
    if displacement_length > max_displacement > 0:
        displacement = scale(displacement, max_displacement / displacement_length)
    return add(linear, scale(displacement, curvature))  # type: ignore[return-value]


def midpoint_uv(a: Sequence[float], b: Sequence[float]) -> tuple[float, float]:
    output = []
    for left, right in zip(a, b):
        if 0 <= left <= 1 and 0 <= right <= 1 and abs(left - right) > 0.5:
            if left < right:
                left += 1
            else:
                right += 1
            output.append(((left + right) * 0.5) % 1.0)
        else:
            output.append((left + right) * 0.5)
    return tuple(output)  # type: ignore[return-value]


def midpoint_skin(
    joints_a: Sequence[int],
    weights_a: Sequence[float],
    joints_b: Sequence[int],
    weights_b: Sequence[float],
) -> tuple[tuple[int, int, int, int], tuple[float, float, float, float]]:
    influences: dict[int, float] = {}
    for joints, weights in ((joints_a, weights_a), (joints_b, weights_b)):
        for joint, weight in zip(joints, weights):
            if weight > 1e-8:
                influences[int(joint)] = influences.get(int(joint), 0.0) + float(weight) * 0.5
    strongest = sorted(influences.items(), key=lambda item: (-item[1], item[0]))[:4]
    total = sum(weight for _, weight in strongest)
    if total <= 1e-12:
        strongest = [(0, 1.0)]
        total = 1.0
    joints = [joint for joint, _ in strongest]
    weights = [weight / total for _, weight in strongest]
    while len(joints) < 4:
        joints.append(0)
        weights.append(0.0)
    return tuple(joints), tuple(weights)  # type: ignore[return-value]


def pack_values(values: Iterable[Sequence], component_type: int, width: int) -> bytes:
    packer = struct.Struct("<" + COMPONENT_FORMAT[component_type] * width)
    output = bytearray()
    for value in values:
        output.extend(packer.pack(*value))
    return bytes(output)


def append_accessor(
    document: dict,
    binary: bytearray,
    values: list[tuple],
    component_type: int,
    accessor_type: str,
    target: int,
    include_bounds: bool = False,
) -> int:
    while len(binary) % 4:
        binary.append(0)
    offset = len(binary)
    width = TYPE_WIDTH[accessor_type]
    packed = pack_values(values, component_type, width)
    binary.extend(packed)
    view_index = len(document["bufferViews"])
    document["bufferViews"].append({
        "buffer": 0,
        "byteLength": len(packed),
        "byteOffset": offset,
        "target": target,
    })
    accessor = {
        "bufferView": view_index,
        "componentType": component_type,
        "count": len(values),
        "type": accessor_type,
    }
    if include_bounds and values:
        accessor["min"] = [min(value[axis] for value in values) for axis in range(width)]
        accessor["max"] = [max(value[axis] for value in values) for axis in range(width)]
    accessor_index = len(document["accessors"])
    document["accessors"].append(accessor)
    return accessor_index


def subdivide_primitive(document: dict, binary: bytearray, primitive: dict, curvature: float) -> dict:
    attributes = primitive["attributes"]
    required = {"POSITION", "NORMAL", "TEXCOORD_0", "JOINTS_0", "WEIGHTS_0"}
    if not required.issubset(attributes):
        raise ValueError(f"skinned subdivision requires {sorted(required)}")
    positions = [tuple(map(float, value)) for value in read_accessor(document, binary, attributes["POSITION"])]
    normals = [normalize(value) for value in read_accessor(document, binary, attributes["NORMAL"])]
    texcoords = [tuple(map(float, value)) for value in read_accessor(document, binary, attributes["TEXCOORD_0"])]
    joints = [tuple(map(int, value)) for value in read_accessor(document, binary, attributes["JOINTS_0"])]
    weights = [tuple(map(float, value)) for value in read_accessor(document, binary, attributes["WEIGHTS_0"])]
    source_indices = [int(value[0]) for value in read_accessor(document, binary, primitive["indices"])]
    if len(source_indices) % 3:
        raise ValueError("triangle index accessor is not divisible by three")

    edge_midpoints: dict[tuple[int, int], int] = {}

    def midpoint_index(left: int, right: int) -> int:
        key = (left, right) if left < right else (right, left)
        existing = edge_midpoints.get(key)
        if existing is not None:
            return existing
        index = len(positions)
        positions.append(curved_midpoint(positions[left], positions[right], normals[left], normals[right], curvature))
        normals.append(normalize(add(normals[left], normals[right])))
        texcoords.append(midpoint_uv(texcoords[left], texcoords[right]))
        midpoint_joints, midpoint_weights = midpoint_skin(joints[left], weights[left], joints[right], weights[right])
        joints.append(midpoint_joints)
        weights.append(midpoint_weights)
        edge_midpoints[key] = index
        return index

    indices: list[tuple[int]] = []
    for offset in range(0, len(source_indices), 3):
        a, b, c = source_indices[offset : offset + 3]
        ab = midpoint_index(a, b)
        bc = midpoint_index(b, c)
        ca = midpoint_index(c, a)
        indices.extend(((a,), (ab,), (ca,), (ab,), (b,), (bc,), (ca,), (bc,), (c,), (ab,), (bc,), (ca,)))

    primitive["attributes"] = {
        **attributes,
        "POSITION": append_accessor(document, binary, positions, 5126, "VEC3", ARRAY_BUFFER, include_bounds=True),
        "NORMAL": append_accessor(document, binary, normals, 5126, "VEC3", ARRAY_BUFFER),
        "TEXCOORD_0": append_accessor(document, binary, texcoords, 5126, "VEC2", ARRAY_BUFFER),
        "JOINTS_0": append_accessor(document, binary, joints, 5121, "VEC4", ARRAY_BUFFER),
        "WEIGHTS_0": append_accessor(document, binary, weights, 5126, "VEC4", ARRAY_BUFFER),
    }
    primitive["indices"] = append_accessor(document, binary, indices, 5125, "SCALAR", ELEMENT_ARRAY_BUFFER)
    primitive.setdefault("extras", {})["weaverSubdivision"] = {
        "method": "weighted-phong-edge",
        "level": 1,
        "curvature": curvature,
        "sourceVertices": len(positions) - len(edge_midpoints),
        "vertices": len(positions),
        "triangles": len(indices) // 3,
    }
    return primitive["extras"]["weaverSubdivision"]


def write_glb(path: Path, document: dict, binary: bytearray) -> None:
    while len(binary) % 4:
        binary.append(0)
    document["buffers"][0]["byteLength"] = len(binary)
    json_payload = json.dumps(document, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
    while len(json_payload) % 4:
        json_payload += b" "
    total_length = 12 + 8 + len(json_payload) + 8 + len(binary)
    output = bytearray(struct.pack("<III", GLB_MAGIC, 2, total_length))
    output.extend(struct.pack("<II", len(json_payload), JSON_CHUNK))
    output.extend(json_payload)
    output.extend(struct.pack("<II", len(binary), BIN_CHUNK))
    output.extend(binary)
    path.write_bytes(output)


def generate_skin_maps(source: Path = SKIN_ALBEDO) -> dict[str, dict]:
    image = Image.open(source).convert("RGB")
    rgb = np.asarray(image, dtype=np.float32) / 255.0
    luminance = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    blurred = np.asarray(
        Image.fromarray(np.uint8(np.clip(luminance, 0, 1) * 255), "L").filter(ImageFilter.GaussianBlur(radius=2.2)),
        dtype=np.float32,
    ) / 255.0
    high_pass = np.clip(luminance - blurred, -0.12, 0.12)
    height, width = luminance.shape
    yy, xx = np.mgrid[0:height, 0:width]
    micro = (
        np.sin(2 * np.pi * (xx * 53 / width + yy * 71 / height)) * 0.0042
        + np.sin(2 * np.pi * (xx * 97 / width - yy * 41 / height)) * 0.0024
    )
    surface = high_pass * 0.72 + micro
    gradient_y, gradient_x = np.gradient(surface)
    normal_x = -gradient_x * 7.5
    normal_y = gradient_y * 7.5
    normal_z = np.ones_like(normal_x)
    length = np.sqrt(normal_x * normal_x + normal_y * normal_y + normal_z * normal_z)
    normal = np.stack((normal_x / length, normal_y / length, normal_z / length), axis=-1)
    normal_rgb = np.uint8(np.clip(normal * 0.5 + 0.5, 0, 1) * 255)

    local_variance = np.abs(high_pass)
    roughness = np.clip(0.55 + (1 - luminance) * 0.105 + local_variance * 0.32, 0.42, 0.72)
    specular = np.clip(0.44 - (roughness - 0.42) * 0.42, 0.24, 0.46)
    specular_alpha = np.uint8(specular * 255)
    specular_rgba = np.empty((height, width, 4), dtype=np.uint8)
    specular_rgba[..., :3] = 255
    specular_rgba[..., 3] = specular_alpha
    maps = {
        "normal": (SKIN_NORMAL, normal_rgb, "RGB"),
        "roughness": (SKIN_ROUGHNESS, np.uint8(roughness * 255), "L"),
        "specular": (SKIN_SPECULAR, specular_rgba, "RGBA"),
    }
    report = {}
    for name, (path, pixels, mode) in maps.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pixels, mode).save(path, format="PNG", optimize=True)
        report[name] = {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "resolution": [width, height],
        }
    return report


def build(source: Path, output: Path, textures: bool = True) -> dict:
    document, binary = read_glb(source)
    mesh_report = {}
    source_triangles = 0
    output_triangles = 0
    for mesh in document.get("meshes", []):
        mesh_name = mesh.get("name", "")
        for primitive in mesh.get("primitives", []):
            index_accessor = document["accessors"][primitive["indices"]]
            source_count = index_accessor["count"] // 3
            source_triangles += source_count
            if mesh_name in SUBDIVISION_PROFILES:
                stats = subdivide_primitive(document, binary, primitive, SUBDIVISION_PROFILES[mesh_name])
                mesh["name"] = f"{mesh_name}_hifi"
                mesh_report[mesh_name] = stats
                output_triangles += stats["triangles"]
            else:
                output_triangles += source_count

    document.setdefault("asset", {})["generator"] = "Weaver HiFi Asset Builder 1.0 + " + document.get("asset", {}).get("generator", "")
    document.setdefault("extras", {})["weaverHighFidelity"] = {
        "version": 1,
        "identity": "original-weaver",
        "source": source.name,
        "method": "skinning-preserving weighted Phong subdivision",
        "sourceTriangles": source_triangles,
        "triangles": output_triangles,
        "meshReport": mesh_report,
    }
    write_glb(output, document, binary)
    texture_report = generate_skin_maps() if textures else {}
    return {
        "source": str(source),
        "output": str(output),
        "sourceBytes": source.stat().st_size,
        "outputBytes": output.stat().st_size,
        "sourceTriangles": source_triangles,
        "triangles": output_triangles,
        "meshes": mesh_report,
        "textures": texture_report,
        "sha256": hashlib.sha256(output.read_bytes()).hexdigest(),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-textures", action="store_true")
    args = parser.parse_args()
    report = build(args.source.resolve(), args.output.resolve(), textures=not args.no_textures)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

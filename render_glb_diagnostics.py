"""Render deterministic front and oblique GLB diagnostics with pyrender."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image


def _render(source: Path, output: Path, yaw_degrees: float) -> None:
    trimesh_scene = trimesh.load(source, force="scene")
    scene = pyrender.Scene.from_trimesh_scene(
        trimesh_scene,
        bg_color=np.array([15, 20, 27, 255]),
        ambient_light=np.array([0.55, 0.55, 0.55]),
    )
    bounds = trimesh_scene.bounds
    center = bounds.mean(axis=0)
    extent = bounds[1] - bounds[0]
    magnitude = float(max(extent[0], extent[1]) * 0.62)
    camera = pyrender.OrthographicCamera(xmag=magnitude, ymag=magnitude)
    pose = trimesh.transformations.rotation_matrix(np.deg2rad(yaw_degrees), [0, 1, 0])
    pose[:3, 3] = center + pose[:3, 2] * float(max(extent) * 3.0)
    scene.add(camera, pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.2), pose=pose)
    renderer = pyrender.OffscreenRenderer(768, 768)
    try:
        color, _ = renderer.render(scene)
    finally:
        renderer.delete()
    Image.fromarray(color).save(output)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("glb", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, yaw in (("front", 0.0), ("left", 42.0), ("right", -42.0)):
        _render(args.glb, args.output_dir / f"diagnostic_{name}.png", yaw)


if __name__ == "__main__":
    main()

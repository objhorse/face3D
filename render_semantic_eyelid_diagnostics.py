"""Render bright, matched-camera pure-geometry eyelid closeups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pyrender
import trimesh
from PIL import Image, ImageDraw


WIDTH = 1100
HEIGHT = 520


def _render(mesh_path: Path, active_vertices: np.ndarray) -> np.ndarray:
    from src.module3_texture import load_mesh_obj

    mesh = trimesh.load(mesh_path, force="mesh", process=False)
    source_vertices, _faces, _uv, _uv_faces = load_mesh_obj(mesh_path)
    active = source_vertices[np.asarray(active_vertices, dtype=np.int64)]
    active_min = active.min(axis=0)
    active_max = active.max(axis=0)
    active_center = (active_min + active_max) * 0.5
    active_extent = active_max - active_min

    material = pyrender.MetallicRoughnessMaterial(
        baseColorFactor=(0.55, 0.62, 0.68, 1.0),
        metallicFactor=0.0,
        roughnessFactor=0.72,
        doubleSided=True,
    )
    render_mesh = pyrender.Mesh.from_trimesh(mesh, material=material, smooth=True)
    scene = pyrender.Scene(
        bg_color=np.array([15, 20, 27, 255]),
        ambient_light=np.array([0.24, 0.24, 0.24]),
    )
    scene.add(render_mesh)

    xmag = max(float(active_extent[0]) * 0.64, 0.06)
    ymag = max(float(active_extent[1]) * 1.9, 0.032)
    camera = pyrender.OrthographicCamera(xmag=xmag, ymag=ymag)
    full_extent = np.asarray(source_vertices.max(axis=0) - source_vertices.min(axis=0))
    camera_pose = np.eye(4)
    camera_pose[:3, 3] = active_center + np.array([0.0, 0.0, max(full_extent) * 3.0])
    scene.add(camera, pose=camera_pose)
    scene.add(
        pyrender.DirectionalLight(color=np.ones(3), intensity=1.8),
        pose=camera_pose,
    )
    fill_pose = camera_pose.copy()
    fill_pose[:3, 3] += np.array([-0.08, 0.05, 0.0])
    scene.add(
        pyrender.DirectionalLight(color=np.array([0.85, 0.92, 1.0]), intensity=0.8),
        pose=fill_pose,
    )
    renderer = pyrender.OffscreenRenderer(WIDTH, HEIGHT)
    try:
        color, _depth = renderer.render(scene)
    finally:
        renderer.delete()
    return color


def render_semantic_eyelid_comparison(
    *,
    baseline_obj: Path,
    candidate_obj: Path,
    controls_json: Path,
    output_dir: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    controls = json.loads(Path(controls_json).read_text(encoding="utf-8"))
    active = np.asarray(controls["active_vertices"], dtype=np.int64)
    baseline = _render(Path(baseline_obj), active)
    candidate = _render(Path(candidate_obj), active)
    Image.fromarray(baseline).save(output_dir / "baseline_eye_closeup.png")
    Image.fromarray(candidate).save(output_dir / "candidate_eye_closeup.png")

    gutter = 18
    header = 48
    comparison = Image.new(
        "RGB",
        (WIDTH * 2 + gutter, HEIGHT + header),
        color=(15, 20, 27),
    )
    comparison.paste(Image.fromarray(baseline), (0, header))
    comparison.paste(Image.fromarray(candidate), (WIDTH + gutter, header))
    draw = ImageDraw.Draw(comparison)
    draw.text((18, 15), "Baseline", fill=(225, 232, 238))
    draw.text((WIDTH + gutter + 18, 15), "Semantic eyelid candidate", fill=(225, 232, 238))
    output_path = output_dir / "eye_geometry_comparison.png"
    comparison.save(output_path)
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline-obj", type=Path, required=True)
    parser.add_argument("--candidate-obj", type=Path, required=True)
    parser.add_argument("--controls-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    render_semantic_eyelid_comparison(
        baseline_obj=args.baseline_obj,
        candidate_obj=args.candidate_obj,
        controls_json=args.controls_json,
        output_dir=args.output_dir,
    )


if __name__ == "__main__":
    main()

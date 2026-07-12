"""Render a textured GLB from the exact reconstruction cameras."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pyrender
import trimesh


def _camera_pose_from_opencv(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    world_to_cv = np.eye(4, dtype=np.float64)
    world_to_cv[:3, :3] = np.asarray(rotation, dtype=np.float64)
    world_to_cv[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    cv_to_gl = np.diag([1.0, -1.0, -1.0, 1.0])
    return np.linalg.inv(cv_to_gl @ world_to_cv)


def _render_view(
    model_path: Path,
    camera: dict,
    size: tuple[int, int],
    clay: bool = False,
) -> tuple[np.ndarray, np.ndarray]:
    width, height = size
    scene = pyrender.Scene(
        bg_color=np.array([16, 20, 26, 0]),
        ambient_light=np.array([0.68, 0.68, 0.68]),
    )
    if clay:
        geometry = trimesh.load(model_path, force="mesh", process=False)
        material = pyrender.MetallicRoughnessMaterial(
            baseColorFactor=(0.66, 0.70, 0.74, 1.0),
            metallicFactor=0.0,
            roughnessFactor=0.88,
        )
        scene.add(pyrender.Mesh.from_trimesh(geometry, material=material, smooth=True))
    else:
        textured_scene = trimesh.load(model_path, force="scene")
        for node_name in textured_scene.graph.nodes_geometry:
            transform, geometry_name = textured_scene.graph[node_name]
            geometry = textured_scene.geometry[geometry_name]
            scene.add(pyrender.Mesh.from_trimesh(geometry, smooth=True), pose=transform)
    intrinsics = np.asarray(camera["K"], dtype=np.float64)
    render_camera = pyrender.IntrinsicsCamera(
        fx=float(intrinsics[0, 0]),
        fy=float(intrinsics[1, 1]),
        cx=float(intrinsics[0, 2]),
        cy=float(intrinsics[1, 2]),
        znear=0.01,
        zfar=5.0,
    )
    pose = _camera_pose_from_opencv(camera["R"], camera["t"])
    scene.add(render_camera, pose=pose)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=1.8), pose=pose)
    renderer = pyrender.OffscreenRenderer(width, height)
    try:
        color, depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
    finally:
        renderer.delete()
    return color, depth


def _label(image: np.ndarray, text: str) -> np.ndarray:
    result = image.copy()
    cv2.rectangle(result, (0, 0), (250, 40), (12, 16, 22), -1)
    cv2.putText(result, text, (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.72, (240, 244, 248), 2, cv2.LINE_AA)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--glb", type=Path, required=True)
    parser.add_argument("--geometry", type=Path)
    parser.add_argument("--cameras", type=Path, required=True)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--calibration", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1024)
    args = parser.parse_args()

    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import _resize_to_target, load_images
    from src.module3_texture import load_cameras

    names = {
        "left": next(args.capture_dir.glob("camera1_*.jpg")).name,
        "front": next(args.capture_dir.glob("camera2_*.jpg")).name,
        "right": next(args.capture_dir.glob("camera3_*.jpg")).name,
    }
    raw = load_images(args.capture_dir, names)
    undistorted, _ = undistort_images_with_calibration(raw, args.calibration, alpha=0.0)
    sources = {view: _resize_to_target(image, args.size) for view, image in undistorted.items()}
    cameras = load_cameras(args.cameras)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for view in ("left", "front", "right"):
        rgba, depth = _render_view(args.glb, cameras[view], (args.size, args.size))
        alpha = rgba[:, :, 3:4].astype(np.float32) / 255.0
        dark = np.full_like(rgba[:, :, :3], [16, 20, 26])
        model = np.clip(rgba[:, :, :3] * alpha + dark * (1.0 - alpha), 0, 255).astype(np.uint8)
        source = sources[view]
        overlay_alpha = (depth > 0).astype(np.float32)[:, :, None] * 0.52
        overlay = np.clip(model * overlay_alpha + source * (1.0 - overlay_alpha), 0, 255).astype(np.uint8)
        panels = [_label(source, f"{view}: source")]
        if args.geometry is not None:
            clay_rgba, _ = _render_view(
                args.geometry, cameras[view], (args.size, args.size), clay=True
            )
            clay_alpha = clay_rgba[:, :, 3:4].astype(np.float32) / 255.0
            clay = np.clip(
                clay_rgba[:, :, :3] * clay_alpha + dark * (1.0 - clay_alpha), 0, 255
            ).astype(np.uint8)
            panels.append(_label(clay, f"{view}: geometry"))
            cv2.imwrite(
                str(args.output_dir / f"{view}_geometry.png"),
                cv2.cvtColor(clay, cv2.COLOR_RGB2BGR),
            )
        panels.extend((_label(model, f"{view}: textured"), _label(overlay, f"{view}: overlay")))
        comparison = np.hstack(panels)
        cv2.imwrite(str(args.output_dir / f"{view}_source.png"), cv2.cvtColor(source, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output_dir / f"{view}_model.png"), cv2.cvtColor(model, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output_dir / f"{view}_overlay.png"), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))
        cv2.imwrite(str(args.output_dir / f"{view}_comparison.jpg"), cv2.cvtColor(comparison, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 94])


if __name__ == "__main__":
    main()

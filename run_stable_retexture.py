"""Rebake a stable geometry output with registered three-view textures."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path


ROOT = Path(__file__).parent
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-output", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    from src import config as cfg
    from src.appearance.stable_texture import run_stable_texture_pipeline
    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import load_images, preprocess_all_views

    args = _parse_args()
    capture_dir = args.capture_dir.resolve()
    source_output = args.source_output.resolve()
    output = args.output.resolve()
    mesh_dir = output / "meshes"
    texture_dir = output / "textures"
    debug_dir = output / "debug"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    for name in (
        "face_mesh.obj",
        "cameras.json",
        "face_stable_neutral.glb",
        "face_stable_geometry.glb",
        "stable_fit_meta.json",
        "stable_semantic_regions.json",
    ):
        source = source_output / "meshes" / name
        if source.exists():
            shutil.copy2(source, mesh_dir / name)

    image_names = {
        "left": next(capture_dir.glob("camera1_*.jpg")).name,
        "front": next(capture_dir.glob("camera2_*.jpg")).name,
        "right": next(capture_dir.glob("camera3_*.jpg")).name,
    }
    raw_images = load_images(capture_dir, image_names)
    hires_images, _ = undistort_images_with_calibration(
        raw_images,
        cfg.CAMERA_CALIBRATION_PATH,
        alpha=cfg.UNDISTORT_ALPHA,
    )
    preprocessed = preprocess_all_views(
        hires_images,
        debug_dir=debug_dir / "preprocess",
        target_size=cfg.WORK_IMAGE_SIZE,
    )
    work_images = {view: data["image"] for view, data in preprocessed.items()}
    face_masks = {view: data["face_mask"] for view, data in preprocessed.items()}

    result = run_stable_texture_pipeline(
        mesh_dir=mesh_dir,
        images=work_images,
        output_texture_dir=texture_dir,
        output_mesh_dir=mesh_dir,
        cfg=cfg,
        preprocessed_views=preprocessed,
        face_masks=face_masks,
        hires_images=hires_images,
        working_image_size=cfg.WORK_IMAGE_SIZE,
    )
    (output / "retexture_result.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    source_viewer = source_output / "model_viewer.html"
    if source_viewer.exists():
        shutil.copy2(source_viewer, output / "model_viewer.html")
    logger.info("Registered texture GLB: %s", result["glb_path"])
    logger.info("Viewer: %s", output / "model_viewer.html")


if __name__ == "__main__":
    main()

"""Rebake frozen geometry with strict projective semantic texture ownership."""

from __future__ import annotations

import argparse
import json
import logging
import shutil
from pathlib import Path

from src.appearance.baseline_texture_lock import BaselineTextureLock, MeshContract
from src.reports.offline_glb_compare import write_offline_glb_compare_viewer


ROOT = Path(__file__).resolve().parent
COPY_ARTIFACTS = (
    "face_mesh.obj",
    "cameras.json",
    "face_stable_neutral.glb",
    "face_stable_geometry.glb",
    "stable_fit_meta.json",
    "stable_semantic_regions.json",
)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args()


def _capture_image_names(capture_dir: Path) -> dict[str, str]:
    result = {}
    for view, camera in (
        ("left", "camera1"),
        ("front", "camera2"),
        ("right", "camera3"),
    ):
        matches = sorted(capture_dir.glob(f"{camera}_*.jpg"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected one {camera} JPG in {capture_dir}, found {len(matches)}"
            )
        result[view] = matches[0].name
    return result


def main() -> None:
    from src import config as cfg
    from src.appearance.stable_texture import run_stable_texture_pipeline
    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import load_images, preprocess_all_views

    args = _parse_args()
    capture_dir = args.capture_dir.resolve()
    baseline = args.baseline.resolve()
    output = args.output.resolve()
    source_mesh_dir = baseline / "meshes"
    mesh_dir = output / "meshes"
    texture_dir = output / "textures"
    debug_dir = output / "debug"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    source_lock = BaselineTextureLock.capture(
        source_mesh_dir,
        ("face_mesh.obj", "cameras.json"),
    )
    source_hashes = source_lock.verify()
    source_contract = MeshContract.from_obj(source_mesh_dir / "face_mesh.obj")
    source_lock.write_manifest(output / "baseline_lock.json")

    for name in COPY_ARTIFACTS:
        source = source_mesh_dir / name
        if not source.is_file():
            raise RuntimeError(f"missing baseline artifact: {source}")
        shutil.copy2(source, mesh_dir / name)
    source_contract.assert_identical(MeshContract.from_obj(mesh_dir / "face_mesh.obj"))

    raw_images = load_images(capture_dir, _capture_image_names(capture_dir))
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
        sampling_mode="strict_projective",
    )

    source_lock.verify()
    source_contract.assert_identical(MeshContract.from_obj(mesh_dir / "face_mesh.obj"))
    copied_lock = BaselineTextureLock.capture(
        mesh_dir,
        ("face_mesh.obj", "cameras.json"),
    )
    copied_lock.assert_hashes(source_hashes)

    baseline_glb = mesh_dir / "face_baseline.glb"
    strict_glb = mesh_dir / "face_strict_projective.glb"
    shutil.copy2(source_mesh_dir / "face.glb", baseline_glb)
    shutil.copy2(mesh_dir / "face.glb", strict_glb)
    result["baseline_lock"] = {
        "verified": True,
        "source_hashes": source_hashes,
        "geometry_contract_identical": True,
        "camera_mapping": {
            "camera1": "left",
            "camera2": "front",
            "camera3": "right",
        },
    }
    result_path = output / "strict_projective_result.json"
    result_path.write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    source_viewer = baseline / "model_viewer.html"
    if source_viewer.is_file():
        shutil.copy2(source_viewer, output / "model_viewer.html")

    dataset_name = capture_dir.name
    viewer = write_offline_glb_compare_viewer(
        output_path=output / "strict_projective_compare.html",
        left_model=baseline_glb,
        right_model=strict_glb,
        vendor_root=ROOT / "frontend" / "vendor",
        title=f"{dataset_name}: same geometry texture comparison",
        left_label="Baseline: registered texture",
        right_label="A3: strict projective + semantic ownership",
    )
    logger.info("Strict GLB: %s", strict_glb)
    logger.info("A/B viewer: %s", viewer)
    logger.info("Result: %s", result_path)


if __name__ == "__main__":
    main()

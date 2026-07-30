"""Re-register texture on the frozen user-approved baseline geometry."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
from pathlib import Path

from src.appearance.baseline_texture_lock import BaselineTextureLock, MeshContract


ROOT = Path(__file__).resolve().parent
BASELINE_NAME = "captures_20260612_135253_controlled_identity_v1"
BASELINE_HASHES = {
    "face_mesh.obj": "ED1CC8B48AFB781788B4D0AC11F5F84F9169AE81CB39E57CA4EF74AE2BE2BF5E",
    "cameras.json": "36F7D0D2AD7E7E8F2B2812EAEB103C26A1A18C569402B21CAC39E28AB596C65C",
    "face.glb": "B17558815F862DA9EE86999551294D082C316BBF8CEE1CD3219638B8AE311860",
}
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
    parser.add_argument(
        "--capture-dir",
        type=Path,
        default=ROOT.parent.parent / "captures_20260612_135253",
    )
    parser.add_argument(
        "--baseline",
        type=Path,
        default=ROOT / "output" / "experiments" / BASELINE_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "output"
            / "experiments"
            / "captures_20260612_135253_baseline_texture_nose_v4"
        ),
    )
    return parser.parse_args()


def _capture_image_names(capture_dir: Path) -> dict[str, str]:
    result = {}
    for view, camera in (("left", "camera1"), ("front", "camera2"), ("right", "camera3")):
        matches = sorted(capture_dir.glob(f"{camera}_*.jpg"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {camera} JPG in {capture_dir}, found {len(matches)}"
            )
        result[view] = matches[0].name
    return result


def _write_embedded_compare_viewer(
    template_path: Path,
    baseline_glb: Path,
    candidate_glb: Path,
    output_path: Path,
) -> Path:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    baseline_data = base64.b64encode(baseline_glb.read_bytes()).decode("ascii")
    candidate_data = base64.b64encode(candidate_glb.read_bytes()).decode("ascii")
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("baseline: '"):
            lines[index] = f"      baseline: '{baseline_data}',"
        elif stripped.startswith("candidate: '"):
            lines[index] = f"      candidate: '{candidate_data}',"
    html = "\n".join(lines)
    html = html.replace("带纹理身份形状对比", "Baseline 纹理语义配准对比")
    html = html.replace("Baseline：原存档几何", "Baseline：认可版本原纹理")
    html = html.replace("Candidate A：受控身份变形", "Registered：同几何语义配准纹理")
    html = html.replace(
        "同一套纹理流程；左为 baseline，右为 Candidate A",
        "左右几何完全相同；只比较五官纹理定位与拼接",
    )
    output_path.write_text(html, encoding="utf-8")
    return output_path


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

    lock = BaselineTextureLock(root=source_mesh_dir, hashes=BASELINE_HASHES)
    verified_hashes = lock.verify()
    baseline_contract = MeshContract.from_obj(source_mesh_dir / "face_mesh.obj")
    lock.write_manifest(output / "baseline_lock.json")
    logger.info("Baseline lock verified: %s", source_mesh_dir)

    for name in COPY_ARTIFACTS:
        source = source_mesh_dir / name
        if not source.exists():
            raise RuntimeError(f"missing required baseline artifact: {source}")
        shutil.copy2(source, mesh_dir / name)
    baseline_contract.assert_identical(MeshContract.from_obj(mesh_dir / "face_mesh.obj"))

    image_names = _capture_image_names(capture_dir)
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

    lock.verify()
    baseline_contract.assert_identical(MeshContract.from_obj(source_mesh_dir / "face_mesh.obj"))
    baseline_contract.assert_identical(MeshContract.from_obj(mesh_dir / "face_mesh.obj"))
    result["baseline_lock"] = {
        "verified": True,
        "source_hashes": verified_hashes,
        "geometry_contract_identical": True,
        "camera_mapping": {"camera1": "left", "camera2": "front", "camera3": "right"},
    }
    (output / "semantic_feature_report.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    template_path = (
        ROOT
        / "output"
        / "experiments"
        / "captures_20260612_135253_controlled_identity_textured_v1"
        / "textured_identity_compare.html"
    )
    viewer_path = output / "baseline_registered_texture_compare.html"
    _write_embedded_compare_viewer(
        template_path,
        source_mesh_dir / "face.glb",
        mesh_dir / "face.glb",
        viewer_path,
    )
    logger.info("Registered texture GLB: %s", mesh_dir / "face.glb")
    logger.info("A/B viewer: %s", viewer_path)


if __name__ == "__main__":
    main()

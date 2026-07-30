"""Rebake three-view texture on the accepted semantic-eyelid geometry."""

from __future__ import annotations

import argparse
import base64
import json
import logging
import shutil
from pathlib import Path

from src.appearance.baseline_texture_lock import (
    BaselineTextureLock,
    MeshContract,
    file_sha256,
)


ROOT = Path(__file__).resolve().parent
BASELINE_NAME = "captures_20260612_135253_controlled_identity_v1"
EYELID_NAME = "captures_20260612_135253_semantic_eyelid_v1"
BASELINE_HASHES = {
    "face_mesh.obj": "ED1CC8B48AFB781788B4D0AC11F5F84F9169AE81CB39E57CA4EF74AE2BE2BF5E",
    "cameras.json": "36F7D0D2AD7E7E8F2B2812EAEB103C26A1A18C569402B21CAC39E28AB596C65C",
    "face.glb": "B17558815F862DA9EE86999551294D082C316BBF8CEE1CD3219638B8AE311860",
}

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
        "--eyelid-experiment",
        type=Path,
        default=ROOT / "output" / "experiments" / EYELID_NAME,
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=(
            ROOT
            / "output"
            / "experiments"
            / "captures_20260612_135253_semantic_eyelid_textured_v1"
        ),
    )
    return parser.parse_args()


def _capture_image_names(capture_dir: Path) -> dict[str, str]:
    names = {}
    for view, camera in (("left", "camera1"), ("front", "camera2"), ("right", "camera3")):
        matches = sorted(capture_dir.glob(f"{camera}_*.jpg"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {camera} JPG in {capture_dir}, found {len(matches)}"
            )
        names[view] = matches[0].name
    return names


def _write_embedded_compare_viewer(
    template_path: Path,
    baseline_glb: Path,
    candidate_glb: Path,
    output_path: Path,
) -> Path:
    lines = template_path.read_text(encoding="utf-8").splitlines()
    embedded = {
        "baseline": base64.b64encode(baseline_glb.read_bytes()).decode("ascii"),
        "candidate": base64.b64encode(candidate_glb.read_bytes()).decode("ascii"),
    }
    for index, line in enumerate(lines):
        stripped = line.strip()
        if stripped.startswith("baseline: '"):
            lines[index] = f"      baseline: '{embedded['baseline']}',"
        elif stripped.startswith("candidate: '"):
            lines[index] = f"      candidate: '{embedded['candidate']}',"
    html = "\n".join(lines)
    html = html.replace("带纹理身份形状对比", "眼睑候选纹理重烘焙对比")
    html = html.replace("Baseline：原存档几何", "Baseline：认可版本原纹理")
    html = html.replace("Candidate A：受控身份变形", "Candidate：眼睑几何 + 新烘焙纹理")
    html = html.replace(
        "同一套纹理流程；左为 baseline，右为 Candidate A",
        "左为认可 baseline；右为眼睑候选与三视角重新烘焙纹理",
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
    eyelid_experiment = args.eyelid_experiment.resolve()
    output = args.output.resolve()
    baseline_mesh_dir = baseline / "meshes"
    candidate_source = eyelid_experiment / "meshes" / "eyelid_candidate.obj"
    mesh_dir = output / "meshes"
    texture_dir = output / "textures"
    debug_dir = output / "debug"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)

    if bool(getattr(cfg, "STABLE_DELETE_INVISIBLE_FACES", False)):
        raise RuntimeError("STABLE_DELETE_INVISIBLE_FACES must be false for this experiment")
    baseline_lock = BaselineTextureLock(
        root=baseline_mesh_dir,
        hashes=BASELINE_HASHES,
    )
    baseline_hashes_before = baseline_lock.verify()
    baseline_lock.write_manifest(output / "baseline_lock.json")
    if not candidate_source.is_file():
        raise RuntimeError(f"missing accepted eyelid candidate: {candidate_source}")
    candidate_hash_before = file_sha256(candidate_source)
    candidate_contract = MeshContract.from_obj(candidate_source)

    shutil.copy2(candidate_source, mesh_dir / "face_mesh.obj")
    for name in (
        "cameras.json",
        "face_stable_neutral.glb",
        "face_stable_geometry.glb",
        "stable_fit_meta.json",
        "stable_semantic_regions.json",
    ):
        source = baseline_mesh_dir / name
        if not source.is_file():
            raise RuntimeError(f"missing baseline artifact: {source}")
        shutil.copy2(source, mesh_dir / name)
    candidate_contract.assert_identical(MeshContract.from_obj(mesh_dir / "face_mesh.obj"))

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
    )

    baseline_hashes_after = baseline_lock.verify()
    candidate_hash_after = file_sha256(candidate_source)
    copied_hash_after = file_sha256(mesh_dir / "face_mesh.obj")
    if candidate_hash_after != candidate_hash_before:
        raise RuntimeError("source eyelid candidate changed during texture baking")
    if copied_hash_after != candidate_hash_before:
        raise RuntimeError("texture pipeline changed the copied candidate OBJ")
    candidate_contract.assert_identical(MeshContract.from_obj(mesh_dir / "face_mesh.obj"))

    result["geometry_lock"] = {
        "candidate_source": str(candidate_source),
        "candidate_sha256": candidate_hash_before,
        "source_unchanged": True,
        "copied_obj_identical": True,
        "geometry_smoothing_on_export": False,
        "delete_invisible_faces": False,
        "contract": {
            "vertices": int(len(candidate_contract.vertices)),
            "faces": int(len(candidate_contract.faces)),
            "uv_vertices": int(len(candidate_contract.uv)),
            "uv_faces": int(len(candidate_contract.uv_faces)),
        },
    }
    result["baseline_lock"] = {
        "verified": baseline_hashes_before == baseline_hashes_after,
        "hashes": baseline_hashes_after,
    }
    report_path = output / "semantic_eyelid_texture_report.json"
    report_path.write_text(
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
    viewer_path = output / "semantic_eyelid_textured_compare.html"
    _write_embedded_compare_viewer(
        template_path,
        baseline_mesh_dir / "face.glb",
        mesh_dir / "face.glb",
        viewer_path,
    )
    logger.info("Candidate texture: %s", texture_dir / "albedo_white.png")
    logger.info("Candidate textured GLB: %s", mesh_dir / "face.glb")
    logger.info("A/B viewer: %s", viewer_path)


if __name__ == "__main__":
    main()

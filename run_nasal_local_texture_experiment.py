"""Rebake only the nasal UV region on the accepted v2 textured baseline."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from run_absolute_eyelid_texture_experiment import (
    _capture_image_names,
    _rig_calibration_from_observation_metadata,
    _write_projection_cameras,
)
from run_multiview_nasal_shape_experiment import (
    _sha256_file,
    _validate_embedded_textured_glb,
    _write_viewer,
    build_model_projection_views,
)
from run_nasal_base_shape_experiment import (
    _load_v4_low_resolution_baseline,
    verify_hash_locked_file,
)
from src.appearance.baseline_texture_lock import MeshContract
from src.reports.nasal_observation_io import load_nasal_observation_bundle


DEFAULT_V2_GLB_SHA256 = (
    "219fde410ddc99abf2367315ffe7407969f17b355e654b74f2a21f0e9fd3c7da"
)
DEFAULT_V2_TEXTURE_SHA256 = (
    "b4402401937e5df8a5dc7fe1755be75c02482424ee807463b613485ca6f8e620"
)
DEFAULT_A2_OBJ_SHA256 = (
    "50051dd20c973ed43f6511eca7b2a5ab4592438901e31da9a74a4ad3bc60201b"
)


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read texture: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write texture: {path}")


def _verify_v2_source(
    source_v2: Path,
    *,
    expected_glb_sha256: str,
    expected_texture_sha256: str,
    expected_obj_sha256: str,
) -> dict[str, Any]:
    source = source_v2.resolve()
    report_path = source / "absolute_eyelid_texture_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing v2 report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "absolute-eyelid-texture-registration-v1"
        or report.get("status") != "success"
    ):
        raise RuntimeError("nasal local texture experiment requires successful v2")
    paths = {
        "glb": source / "meshes" / "face.glb",
        "texture": source / "textures" / "albedo_white.png",
        "obj": source / "meshes" / "face_mesh.obj",
        "cameras": source / "meshes" / "cameras.json",
    }
    hashes = {
        "glb": verify_hash_locked_file(paths["glb"], expected_glb_sha256),
        "texture": verify_hash_locked_file(
            paths["texture"],
            expected_texture_sha256,
        ),
        "obj": verify_hash_locked_file(paths["obj"], expected_obj_sha256),
    }
    source_a2 = Path(report["source_a2"]["path"]).resolve()
    source_a2_report = json.loads(
        (source_a2 / "nasal_base_report.json").read_text(encoding="utf-8")
    )
    return {
        "root": source,
        "report": report,
        "report_path": report_path,
        "paths": paths,
        "hashes": hashes,
        "source_a2": source_a2,
        "source_a2_report": source_a2_report,
    }


def _rename_viewer(path: Path, dataset_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    replacements = {
        "unified multiview nasal shape A/B": (
            "pixel-locked local nasal texture A/B"
        ),
        "Baseline: protected expression depth v3": (
            "Baseline: accepted eye-texture v2"
        ),
        "New: unified multiview nasal shape": (
            "Candidate: same geometry, nasal UV-only registration"
        ),
        "Baseline vs unified nasal shape": (
            "accepted v2 vs nasal UV-only registration"
        ),
        f"{dataset_name}: unified multiview nasal shape A/B": (
            f"{dataset_name}: nasal UV-only texture A/B"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def run_nasal_local_texture_experiment(
    capture_dir: str | Path,
    source_v2: str | Path,
    output: str | Path,
    *,
    expected_v2_glb_sha256: str = DEFAULT_V2_GLB_SHA256,
    expected_v2_texture_sha256: str = DEFAULT_V2_TEXTURE_SHA256,
    expected_a2_obj_sha256: str = DEFAULT_A2_OBJ_SHA256,
    viewer_template: str | Path | None = None,
) -> Path:
    from src import config as cfg
    from src.appearance.nasal_local_texture import (
        build_nasal_uv_alpha,
        composite_pixel_locked_texture,
    )
    from src.appearance.stable_texture import run_stable_texture_pipeline
    from src.geometry.mesh_quality import load_mesh_quality
    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import load_images, preprocess_all_views
    from src.module3_texture import (
        export_glb,
        load_cameras,
        load_mesh_obj,
        transparent_bottom_face_mask,
    )

    captures = Path(capture_dir).resolve()
    target = Path(output).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {target}")
    source = _verify_v2_source(
        Path(source_v2),
        expected_glb_sha256=expected_v2_glb_sha256,
        expected_texture_sha256=expected_v2_texture_sha256,
        expected_obj_sha256=expected_a2_obj_sha256,
    )

    target.mkdir(parents=True)
    mesh_dir = target / "meshes"
    texture_dir = target / "textures"
    debug_dir = target / "debug"
    mesh_dir.mkdir()
    texture_dir.mkdir()
    shutil.copy2(source["paths"]["obj"], mesh_dir / "face_mesh.obj")
    source_geometry = source["source_a2"] / "meshes" / "candidate.glb"
    shutil.copy2(source_geometry, mesh_dir / "face_stable_neutral.glb")
    shutil.copy2(source_geometry, mesh_dir / "face_stable_geometry.glb")
    source_contract = MeshContract.from_obj(source["paths"]["obj"])
    source_contract.assert_identical(
        MeshContract.from_obj(mesh_dir / "face_mesh.obj")
    )

    source_v4 = Path(
        source["source_a2_report"]["source_v4"]["path"]
    ).resolve()
    source_v4_report = json.loads(
        (source_v4 / "fit_report.json").read_text(encoding="utf-8")
    )
    low_resolution_v4 = _load_v4_low_resolution_baseline(
        source_v4,
        source_v4_report,
    )
    observation_path = source_v4 / "nasal_observations.json"
    observation_payload = json.loads(
        observation_path.read_text(encoding="utf-8")
    )
    observations = load_nasal_observation_bundle(observation_path)
    source_work_size = tuple(
        int(value)
        for value in observation_payload.get("metadata", {}).get(
            "work_size_wh",
            (),
        )
    )
    if len(source_work_size) != 2 or min(source_work_size) <= 0:
        raise RuntimeError("nasal observation metadata has no valid work size")
    rig_calibration, rig_hash = _rig_calibration_from_observation_metadata(
        observation_path
    )
    projection_views = build_model_projection_views(
        observations,
        low_resolution_v4.front_rotation,
        low_resolution_v4.front_translation,
    )
    camera_payload = _write_projection_cameras(
        mesh_dir / "cameras.json",
        projection_views,
        source_work_size=source_work_size,
        target_canvas_shape=(cfg.WORK_IMAGE_SIZE, cfg.WORK_IMAGE_SIZE),
    )

    raw_images = load_images(captures, _capture_image_names(captures))
    hires_images, _new_intrinsics = undistort_images_with_calibration(
        raw_images,
        rig_calibration,
        alpha=cfg.UNDISTORT_ALPHA,
    )
    preprocessed = preprocess_all_views(
        hires_images,
        debug_dir=debug_dir / "preprocess",
        target_size=cfg.WORK_IMAGE_SIZE,
    )
    work_images = {
        view: data["image"] for view, data in preprocessed.items()
    }
    face_masks = {
        view: data["face_mask"] for view, data in preprocessed.items()
    }
    vertices, faces, uv_vertices, uv_faces = load_mesh_obj(
        mesh_dir / "face_mesh.obj"
    )
    cameras = load_cameras(mesh_dir / "cameras.json")
    from render_calibrated_model_views import _render_view

    baseline_reference_rgba, _baseline_reference_depth = _render_view(
        source["paths"]["glb"],
        cameras["front"],
        (cfg.WORK_IMAGE_SIZE, cfg.WORK_IMAGE_SIZE),
    )
    baseline_reference_rgb = baseline_reference_rgba[:, :, :3]
    reference_dir = debug_dir / "model_texture_reference"
    reference_dir.mkdir(parents=True, exist_ok=True)
    _write_rgb(
        reference_dir / "front_rendered_v2.png",
        baseline_reference_rgb,
    )

    baseline_quality = load_mesh_quality(
        source_geometry,
        label="accepted_a2_geometry",
    )
    texture_result = run_stable_texture_pipeline(
        mesh_dir=mesh_dir,
        images=work_images,
        output_texture_dir=texture_dir,
        output_mesh_dir=mesh_dir,
        cfg=cfg,
        preprocessed_views=preprocessed,
        face_masks=face_masks,
        hires_images=hires_images,
        working_image_size=cfg.WORK_IMAGE_SIZE,
        baseline_quality=baseline_quality,
        sampling_mode="legacy_registered",
        enable_local_eye_registration=True,
        enable_ordered_nasal_registration=True,
        model_reference_images={"front": baseline_reference_rgb},
    )

    full_candidate_texture_path = texture_dir / "albedo_white.png"
    full_candidate_glb_path = mesh_dir / "face.glb"
    staged_texture_path = texture_dir / "albedo_white_ordered_full.png"
    staged_glb_path = mesh_dir / "face_ordered_full.glb"
    shutil.copy2(full_candidate_texture_path, staged_texture_path)
    shutil.copy2(full_candidate_glb_path, staged_glb_path)

    baseline_texture = _read_rgb(source["paths"]["texture"])
    full_candidate_texture = _read_rgb(staged_texture_path)
    if baseline_texture.shape != full_candidate_texture.shape:
        raise RuntimeError("v2 and ordered candidate texture sizes differ")
    nasal_alpha, ownership_report = build_nasal_uv_alpha(
        vertices,
        faces,
        uv_vertices,
        uv_faces,
        cameras["front"],
        preprocessed["front"]["nose_mask"],
        texture_size=baseline_texture.shape[0],
        source_mask_dilate_px=3,
        feather_px=5.0,
        depth_tolerance_ratio=0.01,
    )
    alpha_path = texture_dir / "nasal_uv_alpha.png"
    ownership_path = texture_dir / "nasal_uv_ownership.png"
    cv2.imwrite(
        str(alpha_path),
        np.rint(nasal_alpha * 255.0).astype(np.uint8),
    )
    cv2.imwrite(
        str(ownership_path),
        (nasal_alpha > 0.0).astype(np.uint8) * 255,
    )
    final_texture, composite_report = composite_pixel_locked_texture(
        baseline_texture,
        full_candidate_texture,
        nasal_alpha,
    )
    _write_rgb(full_candidate_texture_path, final_texture)

    transparent_faces, transparent_y_floor = transparent_bottom_face_mask(
        vertices,
        faces,
        0.05,
    )
    export_glb(
        vertices,
        faces,
        uv_vertices,
        uv_faces,
        final_texture,
        full_candidate_glb_path,
        lighting_type="white",
        lighting_display_name="white",
        smooth_geometry=False,
        transparent_face_mask=transparent_faces,
    )
    shutil.copy2(full_candidate_glb_path, mesh_dir / "face_stable.glb")

    source_contract.assert_identical(
        MeshContract.from_obj(mesh_dir / "face_mesh.obj")
    )
    source_hashes_after = {
        "glb": verify_hash_locked_file(
            source["paths"]["glb"],
            source["hashes"]["glb"],
        ),
        "texture": verify_hash_locked_file(
            source["paths"]["texture"],
            source["hashes"]["texture"],
        ),
        "obj": verify_hash_locked_file(
            source["paths"]["obj"],
            source["hashes"]["obj"],
        ),
    }
    glb_validation = _validate_embedded_textured_glb(
        full_candidate_glb_path
    )
    viewer = _write_viewer(
        template=(
            Path(viewer_template).resolve()
            if viewer_template is not None
            else None
        ),
        baseline_glb=source["paths"]["glb"],
        candidate_glb=full_candidate_glb_path,
        output=target / "nasal_local_texture_compare.html",
        dataset_label=f"{captures.name} | nasal UV-only registration",
    )
    _rename_viewer(viewer, captures.name)

    front_registration = (
        (texture_result.get("registration") or {})
        .get("views", {})
        .get("front", {})
    )
    nose_metrics = (
        front_registration.get("warp_diagnostics", {})
        .get("features", {})
        .get("nose", {})
    )
    report = {
        "schema": "nasal-local-texture-registration-v1",
        "status": "success",
        "dataset": captures.name,
        "source_v2": {
            "path": str(source["root"]),
            "hashes_before": source["hashes"],
            "hashes_after": source_hashes_after,
            "unchanged_after_run": source_hashes_after == source["hashes"],
        },
        "geometry": {
            "selection": "accepted_a2_unchanged",
            "obj_sha256": _sha256_file(mesh_dir / "face_mesh.obj"),
            "topology_unchanged": True,
        },
        "rig": {
            "path": str(rig_calibration),
            "sha256": rig_hash,
        },
        "projection_cameras": camera_payload,
        "registration": {
            "front_nose": nose_metrics,
            "front_warp": front_registration.get("warp_diagnostics", {}),
            "full_texture_pipeline": texture_result,
        },
        "uv_ownership": ownership_report,
        "pixel_locked_composite": composite_report,
        "transparent_bottom": {
            "hidden_faces": int(np.count_nonzero(transparent_faces)),
            "y_floor": float(transparent_y_floor),
        },
        "artifacts": {
            "baseline_glb": str(source["paths"]["glb"]),
            "candidate_glb": str(full_candidate_glb_path),
            "candidate_glb_sha256": _sha256_file(full_candidate_glb_path),
            "candidate_glb_validation": glb_validation,
            "baseline_texture": str(source["paths"]["texture"]),
            "full_ordered_texture": str(staged_texture_path),
            "final_texture": str(full_candidate_texture_path),
            "nasal_uv_alpha": str(alpha_path),
            "nasal_uv_ownership": str(ownership_path),
            "viewer": str(viewer),
        },
    }
    report_path = target / "nasal_local_texture_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Register only nasal UV pixels on accepted v2 texture."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-v2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--expected-v2-glb-sha256",
        default=DEFAULT_V2_GLB_SHA256,
    )
    parser.add_argument(
        "--expected-v2-texture-sha256",
        default=DEFAULT_V2_TEXTURE_SHA256,
    )
    parser.add_argument(
        "--expected-a2-obj-sha256",
        default=DEFAULT_A2_OBJ_SHA256,
    )
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_nasal_local_texture_experiment(
        args.capture_dir,
        args.source_v2,
        args.output,
        expected_v2_glb_sha256=args.expected_v2_glb_sha256,
        expected_v2_texture_sha256=args.expected_v2_texture_sha256,
        expected_a2_obj_sha256=args.expected_a2_obj_sha256,
        viewer_template=args.viewer_template,
    )
    print(f"Nasal local texture report: {report}")


if __name__ == "__main__":
    main()

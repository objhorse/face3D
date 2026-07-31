"""Rebake accepted A2 geometry with independent per-eye texture registration."""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

from run_multiview_nasal_shape_experiment import (
    _load_observation_work_images,
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
from src.geometry.mesh_quality import load_mesh_quality
from src.reports.nasal_observation_io import load_nasal_observation_bundle


DEFAULT_A2_OBJ_SHA256 = (
    "50051dd20c973ed43f6511eca7b2a5ab4592438901e31da9a74a4ad3bc60201b"
)
DEFAULT_A2_GEOMETRY_SHA256 = (
    "780a456a8080b78075e0bcb8f8e382f270b72ef908f9d6592a5610cc0c5a1d99"
)
DEFAULT_A2_TEXTURED_SHA256 = (
    "375d4d84601a5c7fa452991c2dc8b47eb56b44a8bb9478e9114125f627b28328"
)


def _capture_image_names(capture_dir: Path) -> dict[str, str]:
    names = {}
    for view, camera in (
        ("left", "camera1"),
        ("front", "camera2"),
        ("right", "camera3"),
    ):
        matches = sorted(capture_dir.glob(f"{camera}_*.jpg"))
        if len(matches) != 1:
            raise RuntimeError(
                f"expected exactly one {camera} JPG in {capture_dir}, "
                f"found {len(matches)}"
            )
        names[view] = matches[0].name
    return names


def _verify_a2_source(
    source_a2: Path,
    *,
    expected_obj_sha256: str,
    expected_geometry_sha256: str,
    expected_textured_sha256: str,
) -> dict[str, Any]:
    source = source_a2.resolve()
    report_path = source / "nasal_base_report.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing A2 report: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "nasal-base-semantic-a2-v1"
        or report.get("status") != "success"
    ):
        raise RuntimeError("texture experiment requires a successful A2 source")
    paths = {
        "obj": source / "meshes" / "face_mesh.obj",
        "geometry": source / "meshes" / "candidate.glb",
        "textured": source / "meshes" / "candidate_textured.glb",
    }
    hashes = {
        "obj": verify_hash_locked_file(paths["obj"], expected_obj_sha256),
        "geometry": verify_hash_locked_file(
            paths["geometry"],
            expected_geometry_sha256,
        ),
        "textured": verify_hash_locked_file(
            paths["textured"],
            expected_textured_sha256,
        ),
    }
    return {
        "root": source,
        "report_path": report_path,
        "report": report,
        "paths": paths,
        "hashes": hashes,
    }


def _rig_calibration_from_observation_metadata(
    observation_path: Path,
) -> tuple[Path, str]:
    payload = json.loads(observation_path.read_text(encoding="utf-8"))
    rig = payload.get("metadata", {}).get("rig", {})
    calibration_path = Path(str(rig.get("calibration_path", ""))).resolve()
    expected_hash = str(rig.get("sha256", "")).lower()
    if len(expected_hash) != 64:
        raise RuntimeError("observation metadata has no valid rig hash")
    return (
        calibration_path,
        verify_hash_locked_file(calibration_path, expected_hash),
    )


def _write_projection_cameras(
    path: Path,
    projection_views,
    *,
    source_work_size: tuple[int, int],
    target_canvas_shape: tuple[int, int],
) -> dict[str, Any]:
    from src.geometry.observation_coordinates import (
        intrinsics_to_letterbox_canvas,
    )

    semantic_to_texture = {
        "front": "front",
        "subject-left": "left",
        "subject-right": "right",
    }
    views = {}
    for view in projection_views:
        texture_name = semantic_to_texture.get(view.name)
        if texture_name is None:
            raise RuntimeError(f"unexpected projection view: {view.name}")
        canvas_intrinsics = intrinsics_to_letterbox_canvas(
            view.K,
            source_work_size,
            target_canvas_shape,
        )
        views[texture_name] = {
            "K": canvas_intrinsics.astype(float).tolist(),
            "R": view.R_model_to_camera.astype(float).tolist(),
            "t": view.t_model_to_camera.astype(float).tolist(),
        }
    if set(views) != {"left", "front", "right"}:
        raise RuntimeError("projection camera export is missing a required view")
    payload = {
        "schema": "absolute-eyelid-texture-cameras-v1",
        "coordinate_contract": {
            "source_work_size_wh": list(source_work_size),
            "target_canvas_shape_hw": list(target_canvas_shape),
            "mapping": "source_work_pixels_to_square_letterbox_canvas",
        },
        "views": views,
    }
    path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return payload


def _rename_viewer(path: Path, dataset_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "unified multiview nasal shape A/B",
        "independent eyelid texture registration A/B",
    )
    text = text.replace(
        "Baseline: protected expression depth v3",
        "Baseline: accepted A2 original texture",
    )
    text = text.replace(
        "New: unified multiview nasal shape",
        "Candidate: same A2 geometry, independent eye rebake",
    )
    text = text.replace(
        "Baseline vs unified nasal shape",
        "A2 original vs independent eye rebake",
    )
    text = text.replace(
        f"{dataset_name}: unified multiview nasal shape A/B",
        f"{dataset_name}: eyelid texture registration A/B",
    )
    path.write_text(text, encoding="utf-8")


def run_absolute_eyelid_texture_experiment(
    capture_dir: str | Path,
    source_a2: str | Path,
    output: str | Path,
    *,
    expected_obj_sha256: str = DEFAULT_A2_OBJ_SHA256,
    expected_geometry_sha256: str = DEFAULT_A2_GEOMETRY_SHA256,
    expected_textured_sha256: str = DEFAULT_A2_TEXTURED_SHA256,
    viewer_template: str | Path | None = None,
) -> Path:
    from src import config as cfg
    from src.appearance.stable_texture import run_stable_texture_pipeline
    from src.module0_intrinsics import undistort_images_with_calibration
    from src.module1_preprocess import load_images, preprocess_all_views

    captures = Path(capture_dir).resolve()
    target = Path(output).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite experiment output: {target}")
    source = _verify_a2_source(
        Path(source_a2),
        expected_obj_sha256=expected_obj_sha256,
        expected_geometry_sha256=expected_geometry_sha256,
        expected_textured_sha256=expected_textured_sha256,
    )
    target.mkdir(parents=True)
    mesh_dir = target / "meshes"
    texture_dir = target / "textures"
    debug_dir = target / "debug"
    mesh_dir.mkdir()
    texture_dir.mkdir()

    source_contract = MeshContract.from_obj(source["paths"]["obj"])
    copied_obj = mesh_dir / "face_mesh.obj"
    shutil.copy2(source["paths"]["obj"], copied_obj)
    shutil.copy2(
        source["paths"]["geometry"],
        mesh_dir / "face_stable_neutral.glb",
    )
    shutil.copy2(
        source["paths"]["geometry"],
        mesh_dir / "face_stable_geometry.glb",
    )
    source_contract.assert_identical(MeshContract.from_obj(copied_obj))

    source_v4 = Path(source["report"]["source_v4"]["path"]).resolve()
    source_v4_report = json.loads(
        (source_v4 / "fit_report.json").read_text(encoding="utf-8")
    )
    low_resolution_v4 = _load_v4_low_resolution_baseline(
        source_v4,
        source_v4_report,
    )
    observation_path = source_v4 / "nasal_observations.json"
    observations = load_nasal_observation_bundle(observation_path)
    observation_payload = json.loads(
        observation_path.read_text(encoding="utf-8")
    )
    source_work_size = tuple(
        int(value)
        for value in observation_payload.get("metadata", {}).get(
            "work_size_wh",
            (),
        )
    )
    if len(source_work_size) != 2 or min(source_work_size) <= 0:
        raise RuntimeError(
            "nasal observation metadata has no valid work_size_wh"
        )
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
        view: data["image"]
        for view, data in preprocessed.items()
    }
    face_masks = {
        view: data["face_mask"]
        for view, data in preprocessed.items()
    }
    baseline_quality = load_mesh_quality(
        source["paths"]["geometry"],
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
    )
    source_contract.assert_identical(MeshContract.from_obj(copied_obj))
    if _sha256_file(copied_obj) != source["hashes"]["obj"]:
        raise RuntimeError("texture experiment changed the copied A2 OBJ")
    if _sha256_file(source["paths"]["obj"]) != source["hashes"]["obj"]:
        raise RuntimeError("A2 source OBJ changed during texture baking")
    if (
        _sha256_file(source["paths"]["geometry"])
        != source["hashes"]["geometry"]
    ):
        raise RuntimeError("A2 source geometry changed during texture baking")
    if (
        _sha256_file(source["paths"]["textured"])
        != source["hashes"]["textured"]
    ):
        raise RuntimeError("A2 source textured GLB changed during texture baking")

    candidate_glb = mesh_dir / "face.glb"
    glb_validation = _validate_embedded_textured_glb(candidate_glb)
    viewer = _write_viewer(
        template=(
            Path(viewer_template).resolve()
            if viewer_template is not None
            else None
        ),
        baseline_glb=source["paths"]["textured"],
        candidate_glb=candidate_glb,
        output=target / "absolute_eyelid_texture_compare.html",
        dataset_label=f"{captures.name} | eye texture registration",
    )
    _rename_viewer(viewer, captures.name)
    report = {
        "schema": "absolute-eyelid-texture-registration-v1",
        "status": "success",
        "dataset": captures.name,
        "geometry": {
            "selection": "accepted_a2_unchanged",
            "source_obj": str(source["paths"]["obj"]),
            "source_obj_sha256": source["hashes"]["obj"],
            "topology_unchanged": True,
            "reason": (
                "local absolute eye geometry could not reach the full observed "
                "offset without folding the thin orbital surface"
            ),
        },
        "source_a2": {
            "path": str(source["root"]),
            "hashes": source["hashes"],
            "report_sha256": _sha256_file(source["report_path"]),
            "unchanged_after_run": True,
        },
        "rig": {
            "path": str(rig_calibration),
            "sha256": rig_hash,
        },
        "camera_mapping": {
            "camera1": "left",
            "camera2": "front",
            "camera3": "right",
        },
        "projection_cameras": camera_payload,
        "texture": texture_result,
        "artifacts": {
            "baseline_textured_glb": str(source["paths"]["textured"]),
            "candidate_textured_glb": str(candidate_glb),
            "candidate_textured_sha256": _sha256_file(candidate_glb),
            "candidate_glb_validation": glb_validation,
            "texture": str(texture_dir / "albedo_white.png"),
            "viewer": str(viewer),
            "registration_debug": str(
                debug_dir / "stable_texture_registration"
            ),
        },
    }
    report_path = target / "absolute_eyelid_texture_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Rebake A2 with independent left/right eye registration."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-a2", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--expected-obj-sha256", default=DEFAULT_A2_OBJ_SHA256)
    parser.add_argument(
        "--expected-geometry-sha256",
        default=DEFAULT_A2_GEOMETRY_SHA256,
    )
    parser.add_argument(
        "--expected-textured-sha256",
        default=DEFAULT_A2_TEXTURED_SHA256,
    )
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_absolute_eyelid_texture_experiment(
        args.capture_dir,
        args.source_a2,
        args.output,
        expected_obj_sha256=args.expected_obj_sha256,
        expected_geometry_sha256=args.expected_geometry_sha256,
        expected_textured_sha256=args.expected_textured_sha256,
        viewer_template=args.viewer_template,
    )
    print(f"Absolute eyelid texture report: {report}")


if __name__ == "__main__":
    main()

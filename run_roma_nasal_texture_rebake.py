"""Rebake RoMa-guided nasal texture on the immutable v8 candidate geometry."""

from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from run_absolute_eyelid_texture_experiment import _capture_image_names
from run_multiview_nasal_shape_experiment import (
    _sha256_file,
    _validate_embedded_textured_glb,
)
from run_nasal_observation_audit import assert_file_tree_unchanged, file_tree_hashes
from src.appearance.baseline_texture_lock import MeshContract
from src.reports.offline_glb_compare import write_offline_glb_compare_viewer


def _read_rgb(path: Path) -> np.ndarray:
    image = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if image is None:
        raise RuntimeError(f"could not read texture: {path}")
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)


def _write_rgb(path: Path, image: np.ndarray) -> None:
    if not cv2.imwrite(str(path), cv2.cvtColor(image, cv2.COLOR_RGB2BGR)):
        raise RuntimeError(f"could not write texture: {path}")


def _load_v8_source(source_v8: Path) -> dict[str, Any]:
    source = source_v8.resolve()
    report_path = source / "metrics.json"
    if not report_path.is_file():
        raise FileNotFoundError(f"missing v8 metrics: {report_path}")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "nasal-texture-observation-audit-v1"
        or report.get("status") != "ready_for_geometry"
        or not report.get("metadata", {}).get("geometry_modified", False)
    ):
        raise RuntimeError("texture rebake requires successful RoMa v8 geometry")
    artifacts = report.get("metadata", {}).get("candidate_artifacts") or {}
    required = {
        "obj": source / str(artifacts.get("candidate_obj", "")),
        "geometry_glb": source / str(artifacts.get("candidate_geometry_glb", "")),
        "textured_glb": source / str(artifacts.get("candidate_textured_glb", "")),
        "texture": source / str(artifacts.get("texture", "")),
    }
    for label, path in required.items():
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(f"missing v8 {label}: {path}")
    source_v10 = Path(report["metadata"]["source_v10"]).resolve()
    v10_report_path = source_v10 / "alar_surface_report.json"
    v10_report = json.loads(v10_report_path.read_text(encoding="utf-8"))
    source_v5 = Path(v10_report["baseline_contract"]["source_v5"]).resolve()
    v5_report_path = source_v5 / "nasal_local_texture_report.json"
    v5_report = json.loads(v5_report_path.read_text(encoding="utf-8"))
    cameras = source_v5 / "meshes" / "cameras.json"
    rig = Path(v5_report["rig"]["path"]).resolve()
    for label, path in (("cameras", cameras), ("rig", rig)):
        if not path.is_file():
            raise FileNotFoundError(f"missing source {label}: {path}")
    return {
        "root": source,
        "report": report,
        "paths": required,
        "source_v10": source_v10,
        "source_v5": source_v5,
        "cameras": cameras,
        "rig": rig,
        "rig_sha256": _sha256_file(rig),
    }


def run_roma_nasal_texture_rebake(
    capture_dir: str | Path,
    source_v8: str | Path,
    output: str | Path,
    *,
    viewer_vendor_root: str | Path | None = None,
) -> Path:
    from render_calibrated_model_views import _render_view
    from src import config as cfg
    from src.appearance.nasal_local_texture import (
        build_nasal_uv_alpha,
        composite_pixel_locked_texture,
    )
    from src.appearance.roma_texture_controls import build_roma_texture_controls
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
    source = _load_v8_source(Path(source_v8))
    source_hashes = file_tree_hashes(source["root"])
    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{target.name}.partial-", dir=str(target.parent))
    )
    try:
        mesh_dir = staging / "meshes"
        texture_dir = staging / "textures"
        debug_dir = staging / "debug"
        mesh_dir.mkdir()
        texture_dir.mkdir()
        candidate_obj = mesh_dir / "face_mesh.obj"
        shutil.copy2(source["paths"]["obj"], candidate_obj)
        shutil.copy2(source["cameras"], mesh_dir / "cameras.json")
        for name in ("face_stable_neutral.glb", "face_stable_geometry.glb"):
            shutil.copy2(source["paths"]["geometry_glb"], mesh_dir / name)
        source_contract = MeshContract.from_obj(source["paths"]["obj"])
        source_contract.assert_identical(MeshContract.from_obj(candidate_obj))

        raw_images = load_images(captures, _capture_image_names(captures))
        hires_images, _new_intrinsics = undistort_images_with_calibration(
            raw_images,
            source["rig"],
            alpha=cfg.UNDISTORT_ALPHA,
        )
        preprocessed = preprocess_all_views(
            hires_images,
            debug_dir=debug_dir / "preprocess",
            target_size=cfg.WORK_IMAGE_SIZE,
        )
        work_images = {view: data["image"] for view, data in preprocessed.items()}
        face_masks = {view: data["face_mask"] for view, data in preprocessed.items()}
        vertices, faces, uv_vertices, uv_faces = load_mesh_obj(candidate_obj)
        cameras = load_cameras(mesh_dir / "cameras.json")
        roma_controls = build_roma_texture_controls(
            source["report"],
            vertices,
            faces,
            cameras,
            work_size=(640, 480),
            canvas_shape=(cfg.WORK_IMAGE_SIZE, cfg.WORK_IMAGE_SIZE),
        )
        if any(
            view not in roma_controls or len(roma_controls[view].model_points) < 4
            for view in ("left", "front", "right")
        ):
            raise RuntimeError("RoMa controls do not cover all three texture views")

        baseline_rgba, _baseline_depth = _render_view(
            source["paths"]["textured_glb"],
            cameras["front"],
            (cfg.WORK_IMAGE_SIZE, cfg.WORK_IMAGE_SIZE),
        )
        baseline_reference = baseline_rgba[:, :, :3]
        reference_dir = debug_dir / "model_texture_reference"
        reference_dir.mkdir(parents=True)
        _write_rgb(reference_dir / "front_rendered_v8.png", baseline_reference)

        baseline_quality = load_mesh_quality(
            source["paths"]["geometry_glb"],
            label="roma_v8_geometry",
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
            enable_local_eye_registration=False,
            enable_ordered_nasal_registration=True,
            model_reference_images={"front": baseline_reference},
            external_nasal_controls_by_view=roma_controls,
        )

        final_texture_path = texture_dir / "albedo_white.png"
        full_texture_path = texture_dir / "albedo_white_roma_full.png"
        full_glb_path = mesh_dir / "face_roma_full.glb"
        shutil.copy2(final_texture_path, full_texture_path)
        shutil.copy2(mesh_dir / "face.glb", full_glb_path)
        baseline_texture = _read_rgb(source["paths"]["texture"])
        full_texture = _read_rgb(full_texture_path)
        if baseline_texture.shape != full_texture.shape:
            raise RuntimeError("v8 and RoMa rebake texture sizes differ")
        nasal_alpha, ownership_report = build_nasal_uv_alpha(
            vertices,
            faces,
            uv_vertices,
            uv_faces,
            cameras["front"],
            preprocessed["front"]["nose_mask"],
            texture_size=baseline_texture.shape[0],
            source_mask_dilate_px=6,
            feather_px=12.0,
            depth_tolerance_ratio=0.01,
        )
        alpha_path = texture_dir / "nasal_uv_alpha.png"
        cv2.imwrite(str(alpha_path), np.rint(nasal_alpha * 255.0).astype(np.uint8))
        final_texture, composite_report = composite_pixel_locked_texture(
            baseline_texture,
            full_texture,
            nasal_alpha,
        )
        _write_rgb(final_texture_path, final_texture)
        outside = nasal_alpha <= 0.0
        if not np.array_equal(final_texture[outside], baseline_texture[outside]):
            raise RuntimeError("nasal composite changed pixels outside UV ownership")

        transparent_faces, transparent_y_floor = transparent_bottom_face_mask(
            vertices,
            faces,
            0.05,
        )
        candidate_glb = mesh_dir / "candidate_textured_v9.glb"
        export_glb(
            vertices,
            faces,
            uv_vertices,
            uv_faces,
            final_texture,
            candidate_glb,
            lighting_type="white",
            lighting_display_name="white",
            smooth_geometry=False,
            transparent_face_mask=transparent_faces,
        )
        shutil.copy2(candidate_glb, mesh_dir / "face.glb")
        glb_validation = dict(_validate_embedded_textured_glb(candidate_glb))
        glb_validation["path"] = "meshes/candidate_textured_v9.glb"
        glb_validation["valid"] = True
        source_contract.assert_identical(MeshContract.from_obj(candidate_obj))
        assert_file_tree_unchanged(source["root"], source_hashes)

        viewer = write_offline_glb_compare_viewer(
            output_path=staging / "roma_nasal_texture_compare.html",
            left_model=source["paths"]["textured_glb"],
            right_model=candidate_glb,
            vendor_root=(
                Path(viewer_vendor_root).resolve()
                if viewer_vendor_root is not None
                else Path(__file__).resolve().parent / "frontend" / "vendor"
            ),
            title=f"{captures.name}: RoMa v8 geometry texture rebake",
            left_label="Baseline: v8 geometry with inherited texture",
            right_label="Candidate: same v8 geometry with RoMa-guided nasal rebake",
        )
        report = {
            "schema": "roma-nasal-texture-rebake-v1",
            "status": "success",
            "dataset": captures.name,
            "source_v8": {
                "path": str(source["root"]),
                "unchanged_after_run": True,
                "candidate_obj_sha256": _sha256_file(source["paths"]["obj"]),
                "candidate_glb_sha256": _sha256_file(source["paths"]["textured_glb"]),
                "texture_sha256": _sha256_file(source["paths"]["texture"]),
            },
            "geometry": {
                "selection": "roma_v8_candidate_unchanged",
                "obj_sha256": _sha256_file(candidate_obj),
                "topology_and_uv_unchanged": True,
            },
            "rig": {"path": str(source["rig"]), "sha256": source["rig_sha256"]},
            "roma_texture_controls": {
                view: {
                    "count": int(len(controls.model_points)),
                    "metadata": dict(controls.metadata),
                }
                for view, controls in roma_controls.items()
            },
            "texture_pipeline": texture_result,
            "uv_ownership": ownership_report,
            "pixel_locked_composite": composite_report,
            "transparent_bottom": {
                "hidden_faces": int(np.count_nonzero(transparent_faces)),
                "y_floor": float(transparent_y_floor),
            },
            "artifacts": {
                "baseline_glb": str(source["paths"]["textured_glb"]),
                "candidate_glb": "meshes/candidate_textured_v9.glb",
                "candidate_glb_validation": glb_validation,
                "baseline_texture": str(source["paths"]["texture"]),
                "full_rebake_texture": "textures/albedo_white_roma_full.png",
                "final_texture": "textures/albedo_white.png",
                "nasal_uv_alpha": "textures/nasal_uv_alpha.png",
                "viewer": "roma_nasal_texture_compare.html",
            },
        }
        report_path = staging / "roma_nasal_texture_report.json"
        report_path.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        relative_report = report_path.relative_to(staging)
        staging.rename(target)
        return target / relative_report
    finally:
        if staging.exists():
            shutil.rmtree(staging)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-v8", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--viewer-vendor-root", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_roma_nasal_texture_rebake(
        args.capture_dir,
        args.source_v8,
        args.output,
        viewer_vendor_root=args.viewer_vendor_root,
    )
    print(report)


if __name__ == "__main__":
    main()

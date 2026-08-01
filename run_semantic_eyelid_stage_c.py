"""Run Stage C semantic eyelid refinement from an immutable approved A2 model."""

from __future__ import annotations

import argparse
import html
import json
import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from run_expression_depth_experiment import _export_with_baseline_texture
from run_multiview_nasal_shape_experiment import (
    _load_observation_work_images,
    _sha256_file,
    _validate_embedded_textured_glb,
    _write_viewer,
    build_model_projection_views,
)
from run_nasal_base_shape_experiment import (
    _detect_work_landmarks,
    _load_v4_low_resolution_baseline,
    verify_hash_locked_file,
)
from src.geometry.eye_state import aggregate_eye_state, eye_view_evidence
from src.geometry.eyelid_fit import project_landmarks
from src.geometry.semantic_eyelid_optimizer import (
    SemanticEyelidOptimizationConfig,
    build_view_eye_weights,
    fit_semantic_eyelids_stage_c,
)
from src.geometry.semantic_eyelid_rig import build_semantic_eyelid_rig
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
DEFAULT_A2_TEXTURE_SHA256 = (
    "0b9302581972a9cb7bf7947efb1f9f7e8ec06c08170ccc5371c25b3cc6fd262b"
)


def verify_stage_c_source(
    source_a2: str | Path,
    *,
    expected_obj_sha256: str = DEFAULT_A2_OBJ_SHA256,
    expected_geometry_sha256: str = DEFAULT_A2_GEOMETRY_SHA256,
    expected_textured_sha256: str = DEFAULT_A2_TEXTURED_SHA256,
    expected_texture_sha256: str = DEFAULT_A2_TEXTURE_SHA256,
) -> dict[str, Any]:
    source = Path(source_a2).resolve()
    if not source.is_dir():
        raise FileNotFoundError(f"A2 source output does not exist: {source}")
    report_path = source / "nasal_base_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if (
        report.get("schema") != "nasal-base-semantic-a2-v1"
        or report.get("status") != "success"
    ):
        raise RuntimeError("Stage C source must be a successful A2 experiment")
    paths = {
        "obj": source / "meshes" / "face_mesh.obj",
        "geometry": source / "meshes" / "candidate.glb",
        "textured": source / "meshes" / "candidate_textured.glb",
        "texture": source / "textures" / "albedo_baseline_locked.png",
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
        "texture": verify_hash_locked_file(
            paths["texture"],
            expected_texture_sha256,
        ),
    }
    return {
        "source": source,
        "report_path": report_path,
        "report": report,
        "paths": paths,
        "hashes": hashes,
    }


def _rig_calibration_from_observation_metadata(
    observation_path: Path,
) -> tuple[Path, str]:
    payload = json.loads(observation_path.read_text(encoding="utf-8"))
    metadata = payload.get("metadata", {})
    rig = metadata.get("rig", {})
    calibration_path = Path(str(rig.get("calibration_path", ""))).resolve()
    expected_hash = str(rig.get("sha256", "")).lower()
    if len(expected_hash) != 64:
        raise RuntimeError("nasal observation metadata has no valid rig hash")
    actual_hash = verify_hash_locked_file(calibration_path, expected_hash)
    return calibration_path, actual_hash


def _projection_cameras(views) -> dict[str, dict[str, np.ndarray]]:
    return {
        view.name: {
            "K": np.asarray(view.K, dtype=np.float64),
            "R": np.asarray(view.R_model_to_camera, dtype=np.float64),
            "t": np.asarray(view.t_model_to_camera, dtype=np.float64),
        }
        for view in views
    }


def _write_eye_overlay(
    image: np.ndarray,
    observed: np.ndarray,
    baseline: np.ndarray,
    candidate: np.ndarray,
    output_path: Path,
) -> None:
    canvas = np.asarray(image).copy()
    eye_paths = ((36, 37, 38, 39, 40, 41), (42, 43, 44, 45, 46, 47))
    for label, points, color in (
        ("observed", observed, (215, 65, 215)),
        ("A2", baseline, (35, 210, 255)),
        ("Stage C", candidate, (70, 235, 80)),
    ):
        for indices in eye_paths:
            polygon = np.rint(points[np.asarray(indices)]).astype(np.int32)
            cv2.polylines(canvas, [polygon], True, color, 2, cv2.LINE_AA)
            for point in polygon:
                cv2.circle(canvas, tuple(point), 3, color, -1, cv2.LINE_AA)
        x = 22 + (0 if label == "observed" else 145 if label == "A2" else 230)
        cv2.putText(
            canvas,
            label,
            (x, 34),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.68,
            color,
            2,
            cv2.LINE_AA,
        )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(output_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def _rename_viewer(path: Path, dataset_name: str) -> None:
    text = path.read_text(encoding="utf-8")
    label = f"{dataset_name} | semantic eyelid Stage C"
    replacements = {
        (
            f"<title>{label} | Baseline vs unified nasal shape</title>"
        ): f"<title>{label} | accepted A2 vs fixed-corner eyelids</title>",
        (
            f'<div id="baseline-label" class="label">{label} | '
            "Baseline: protected expression depth v3</div>"
        ): (
            f'<div id="baseline-label" class="label">{label} | '
            "accepted A2</div>"
        ),
        (
            f'<div id="candidate-label" class="label">{label} | '
            "New: unified multiview nasal shape</div>"
        ): (
            f'<div id="candidate-label" class="label">{label} | '
            "fixed-corner eyelid fit</div>"
        ),
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    path.write_text(text, encoding="utf-8")


def _write_diagnostic_index(
    output: Path,
    report: Mapping[str, Any],
) -> Path:
    rows = []
    before = report["optimization"]["before"]["by_view_eye_px"]
    after = report["optimization"]["after"]["by_view_eye_px"]
    for view_name in ("front", "subject-left", "subject-right"):
        for eye_name in ("subject_right", "subject_left"):
            rows.append(
                "<tr>"
                f"<td>{html.escape(view_name)}</td>"
                f"<td>{html.escape(eye_name)}</td>"
                f"<td>{before[view_name][eye_name]:.3f}</td>"
                f"<td>{after[view_name][eye_name]:.3f}</td>"
                "</tr>"
            )
    cards = []
    for view_name in ("front", "subject-left", "subject-right"):
        image = f"projection_overlays/{view_name}.png"
        cards.append(
            "<figure>"
            f"<img src=\"{html.escape(image)}\" alt=\"{html.escape(view_name)}\">"
            f"<figcaption>{html.escape(view_name)}</figcaption>"
            "</figure>"
        )
    document = f"""<!doctype html>
<html lang="zh-CN">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Stage C eyelid diagnostics</title>
  <style>
    body {{ margin: 0; background: #10151b; color: #edf3f7; font: 15px/1.5 Arial, sans-serif; }}
    main {{ max-width: 1320px; margin: 0 auto; padding: 24px; }}
    h1 {{ font-size: 25px; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(320px, 1fr)); gap: 12px; }}
    figure {{ margin: 0; border: 1px solid #35434f; background: #18212a; }}
    img {{ width: 100%; display: block; }}
    figcaption {{ padding: 8px 10px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 20px; }}
    th, td {{ border: 1px solid #35434f; padding: 8px; text-align: left; }}
    th {{ background: #202c36; }}
  </style>
</head>
<body><main>
  <h1>Stage C 眼睑几何诊断</h1>
  <p>紫色：观测；黄色：已验收 A2；绿色：Stage C。纹理不参与几何评分。</p>
  <div class="grid">{''.join(cards)}</div>
  <table>
    <thead><tr><th>视角</th><th>眼睛</th><th>A2 / px</th><th>Stage C / px</th></tr></thead>
    <tbody>{''.join(rows)}</tbody>
  </table>
</main></body></html>"""
    path = output / "debug" / "eyelid_observations" / "index.html"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(document, encoding="utf-8")
    return path


def run_semantic_eyelid_stage_c(
    capture_dir: str | Path,
    source_a2: str | Path,
    output: str | Path,
    *,
    expected_obj_sha256: str = DEFAULT_A2_OBJ_SHA256,
    expected_geometry_sha256: str = DEFAULT_A2_GEOMETRY_SHA256,
    expected_textured_sha256: str = DEFAULT_A2_TEXTURED_SHA256,
    expected_texture_sha256: str = DEFAULT_A2_TEXTURE_SHA256,
    support_rings: int = 14,
    viewer_template: str | Path | None = None,
) -> Path:
    """Optimize only semantic eyelids and preserve the approved A2 elsewhere."""
    captures = Path(capture_dir).resolve()
    target = Path(output).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite Stage C output: {target}")
    source = verify_stage_c_source(
        source_a2,
        expected_obj_sha256=expected_obj_sha256,
        expected_geometry_sha256=expected_geometry_sha256,
        expected_textured_sha256=expected_textured_sha256,
        expected_texture_sha256=expected_texture_sha256,
    )
    target.mkdir(parents=True)

    from src import config as cfg
    from src.module2_geometry import (
        FLAMEModel,
        _mediapipe_to_68,
        export_mesh_glb,
        export_mesh_obj,
        load_flame_landmark_mapping,
    )
    from src.module3_texture import load_mesh_obj

    vertices, faces, uv, uv_faces = load_mesh_obj(source["paths"]["obj"])
    flame = FLAMEModel(cfg.FLAME_MODEL_PATH, n_shape=1, n_exp=1)
    mapping = load_flame_landmark_mapping(cfg.FLAME_LANDMARK_PATH)
    if mapping is None:
        raise RuntimeError("FLAME landmark embedding is required")
    flame_faces = flame.faces.detach().cpu().numpy()
    landmark_triangles = flame_faces[
        np.asarray(mapping["face_idx"], dtype=np.int64)
    ]
    barycentric = np.asarray(mapping["bary_coords"], dtype=np.float64)
    if int(landmark_triangles.max()) >= len(vertices):
        raise RuntimeError("A2 mesh no longer preserves FLAME source vertex ordering")

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
    rig_calibration, rig_hash = _rig_calibration_from_observation_metadata(
        observation_path
    )
    work_images = _load_observation_work_images(
        observation_path,
        observations,
        rig_calibration,
    )
    dense_landmarks = _detect_work_landmarks(work_images)
    observed = {
        view_name: _mediapipe_to_68(points)
        for view_name, points in dense_landmarks.items()
    }
    views = build_model_projection_views(
        observations,
        low_resolution_v4.front_rotation,
        low_resolution_v4.front_translation,
    )
    cameras = _projection_cameras(views)
    evidence = {
        view_name: eye_view_evidence(points)
        for view_name, points in dense_landmarks.items()
    }
    consensus = aggregate_eye_state(evidence)
    view_eye_weights = build_view_eye_weights(evidence, consensus)

    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        landmark_triangles,
        support_rings=int(support_rings),
        core_rings=2,
        sigma_rings=4.0,
        smoothing_iterations=80,
        smoothing_retention=0.02,
        freeze_corner_seeds=True,
    )
    result = fit_semantic_eyelids_stage_c(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=landmark_triangles,
        barycentric=barycentric,
        cameras=cameras,
        observed_landmarks=observed,
        eye_states=consensus.states,
        view_eye_weights=view_eye_weights,
        cfg=SemanticEyelidOptimizationConfig(),
    )

    debug_dir = target / "debug"
    overlay_dir = debug_dir / "eyelid_observations" / "projection_overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    for view_name, camera in cameras.items():
        baseline_projection = project_landmarks(
            vertices,
            landmark_triangles,
            barycentric,
            camera,
        )
        candidate_projection = project_landmarks(
            result.vertices,
            landmark_triangles,
            barycentric,
            camera,
        )
        _write_eye_overlay(
            work_images[view_name],
            observed[view_name],
            baseline_projection,
            candidate_projection,
            overlay_dir / f"{view_name}.png",
        )

    mesh_dir = target / "meshes"
    texture_dir = target / "textures"
    mesh_dir.mkdir(parents=True, exist_ok=True)
    texture_dir.mkdir(parents=True, exist_ok=True)
    baseline_geometry = mesh_dir / "baseline_a2_geometry.glb"
    baseline_textured = mesh_dir / "baseline_a2_textured.glb"
    shutil.copy2(source["paths"]["geometry"], baseline_geometry)
    shutil.copy2(source["paths"]["textured"], baseline_textured)
    candidate_obj = mesh_dir / "face_mesh.obj"
    candidate_geometry = mesh_dir / "candidate.glb"
    candidate_textured = mesh_dir / "candidate_textured.glb"
    export_mesh_obj(
        result.vertices,
        faces,
        uv,
        uv_faces,
        candidate_obj,
    )
    export_mesh_glb(
        result.vertices,
        faces,
        uv,
        uv_faces,
        candidate_geometry,
    )
    texture_path = texture_dir / "albedo_baseline_locked.png"
    shutil.copy2(source["paths"]["texture"], texture_path)
    if _sha256_file(texture_path) != source["hashes"]["texture"]:
        raise RuntimeError("Stage C texture copy changed the locked texture")
    _export_with_baseline_texture(
        mesh_path=candidate_obj,
        texture_path=texture_path,
        output_path=candidate_textured,
    )
    glb_validation = _validate_embedded_textured_glb(candidate_textured)
    if _sha256_file(source["paths"]["textured"]) != source["hashes"]["textured"]:
        raise RuntimeError("approved A2 source changed during Stage C")

    viewer = _write_viewer(
        template=Path(viewer_template).resolve() if viewer_template else None,
        baseline_glb=baseline_textured,
        candidate_glb=candidate_textured,
        output=target / "semantic_eyelid_stage_c_compare.html",
        dataset_label=f"{captures.name} | semantic eyelid Stage C",
    )
    _rename_viewer(viewer, captures.name)
    report = {
        "schema": "semantic-eyelid-stage-c-v1",
        "status": "success" if result.report["success"] else "diagnostic",
        "dataset": captures.name,
        "texture_scoring_in_objective": False,
        "source_a2": {
            "path": str(source["source"]),
            "hashes": source["hashes"],
            "source_report_sha256": _sha256_file(source["report_path"]),
            "unchanged_after_run": True,
        },
        "rig": {
            "path": str(rig_calibration),
            "sha256": rig_hash,
        },
        "eye_state": consensus.to_dict(),
        "view_eye_weights": view_eye_weights,
        "semantic_rig": rig.to_dict(),
        "optimization": result.report,
        "artifacts": {
            "baseline_geometry_glb": str(baseline_geometry),
            "baseline_textured_glb": str(baseline_textured),
            "candidate_obj": str(candidate_obj),
            "candidate_geometry_glb": str(candidate_geometry),
            "candidate_geometry_sha256": _sha256_file(candidate_geometry),
            "candidate_textured_glb": str(candidate_textured),
            "candidate_textured_sha256": _sha256_file(candidate_textured),
            "locked_texture": str(texture_path),
            "locked_texture_sha256": _sha256_file(texture_path),
            "viewer": str(viewer),
            "glb_validation": glb_validation,
        },
    }
    diagnostic_index = _write_diagnostic_index(target, report)
    report["artifacts"]["diagnostic_index"] = str(diagnostic_index)
    report_path = target / "stage_c_report.json"
    report_path.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return report_path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run fixed-corner Stage C semantic eyelid refinement."
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
    parser.add_argument(
        "--expected-texture-sha256",
        default=DEFAULT_A2_TEXTURE_SHA256,
    )
    parser.add_argument("--support-rings", type=int, default=14)
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_semantic_eyelid_stage_c(
        args.capture_dir,
        args.source_a2,
        args.output,
        expected_obj_sha256=args.expected_obj_sha256,
        expected_geometry_sha256=args.expected_geometry_sha256,
        expected_textured_sha256=args.expected_textured_sha256,
        expected_texture_sha256=args.expected_texture_sha256,
        support_rings=args.support_rings,
        viewer_template=args.viewer_template,
    )
    print(f"Stage C report: {report}")


if __name__ == "__main__":
    main()

"""Run A2 nasal-base refinement from the immutable balanced v4 result."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import html
import json
import os
import shutil
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from run_multiview_nasal_shape_experiment import (
    _export_candidate,
    _load_baseline_state,
    _load_observation_work_images,
    _sha256_file,
    _write_viewer,
    build_model_projection_views,
)
from src.geometry.nasal_base_observations import (
    NasalBaseObservationBundle,
    build_nasal_base_observations,
)
from src.geometry.nasal_base_optimizer import (
    NasalBaseOptimizationContext,
    fit_staged_nasal_base,
    project_nasal_base_landmarks,
)
from src.geometry.nasal_base_semantic_basis import (
    build_nasal_base_semantic_basis,
)
from src.geometry.nasal_semantic_basis import (
    apply_nasal_semantic_basis,
    build_nasal_semantic_basis,
)
from src.geometry.rig_consistent_pose import (
    RigConsistentPoseContext,
    fit_rig_consistent_pose,
)
from src.module1_preprocess import detect_landmarks_mediapipe
from src.reports.nasal_geometry_report import validate_minimal_nasal_candidate
from src.reports.nasal_observation_io import load_nasal_observation_bundle


DEFAULT_V4_GEOMETRY_SHA256 = (
    "b3ca8484d33a3c453fd975de1b09adad6b782e6686bd965ddb78da58ad95f6f6"
)
DEFAULT_V4_TEXTURED_SHA256 = (
    "bc610fc635bbce9f12d90c87f60c5a5d97d04217016e29edd6ff13b24d1ade69"
)

__all__ = [
    "DEFAULT_V4_GEOMETRY_SHA256",
    "DEFAULT_V4_TEXTURED_SHA256",
    "run_nasal_base_shape_experiment",
    "verify_hash_locked_file",
]


def verify_hash_locked_file(path: str | Path, expected_sha256: str) -> str:
    artifact = Path(path).resolve()
    if not artifact.is_file() or artifact.stat().st_size <= 0:
        raise FileNotFoundError(f"hash-locked artifact does not exist: {artifact}")
    expected = str(expected_sha256)
    if (
        len(expected) != 64
        or any(character not in "0123456789abcdef" for character in expected)
    ):
        raise ValueError("expected_sha256 must be 64 lowercase hex digits")
    actual = _sha256_file(artifact)
    if actual != expected:
        raise RuntimeError(
            f"hash-locked artifact hash mismatch: expected {expected}, got {actual}"
        )
    return actual


def _copy_new(source: Path, destination: Path) -> str:
    if destination.exists():
        raise FileExistsError(f"refusing to overwrite artifact: {destination}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    source_hash = _sha256_file(source)
    copied_hash = _sha256_file(destination)
    if copied_hash != source_hash:
        raise RuntimeError(f"copied artifact hash mismatch: {destination}")
    return copied_hash


def _load_v4_low_resolution_baseline(
    source_v4: Path,
    report: Mapping[str, Any],
):
    protected_source = Path(report["paths"]["source_output"]).resolve()
    baseline = _load_baseline_state(protected_source)
    ordering = tuple(report["fit"]["parameterization"]["ordering"])
    coefficients = np.asarray(
        report["fit"]["parameterization"]["coefficients"],
        dtype=np.float64,
    )
    semantic_basis = build_nasal_semantic_basis(
        baseline.vertices,
        baseline.faces,
        baseline.landmark_triangles,
        baseline.landmark_barycentric,
        baseline.front_rotation,
    )
    if ordering != semantic_basis.names:
        raise RuntimeError("v4 report semantic parameter ordering is incompatible")
    v4_vertices, _displacement = apply_nasal_semantic_basis(
        baseline.vertices,
        semantic_basis,
        coefficients,
    )
    if not np.isfinite(v4_vertices).all():
        raise RuntimeError("reconstructed v4 low-resolution vertices are non-finite")
    return dataclasses.replace(
        baseline,
        vertices=np.asarray(v4_vertices, dtype=np.float64),
    )


def _detect_work_landmarks(
    work_images: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    landmarks = {}
    for semantic_view in ("front", "subject-left", "subject-right"):
        points, _visibility = detect_landmarks_mediapipe(
            np.asarray(work_images[semantic_view]),
            view_name=f"A2 {semantic_view}",
        )
        if points is None or np.asarray(points).shape != (478, 2):
            raise RuntimeError(
                f"MediaPipe failed to produce 478 points for {semantic_view}"
            )
        landmarks[semantic_view] = np.asarray(points, dtype=np.float64)
    return landmarks


def _projection_metrics(
    vertices: np.ndarray,
    context: NasalBaseOptimizationContext,
) -> dict[str, Any]:
    result = {}
    for semantic_view, observation in context.observations.by_view.items():
        projected = project_nasal_base_landmarks(
            vertices,
            context.landmark_triangles,
            context.landmark_barycentric,
            context.views_by_name[semantic_view],
        )
        local = projected[observation.landmark_68_indices - 31]
        errors = np.linalg.norm(local - observation.target_xy, axis=1)
        result[semantic_view] = {
            "mean_px": float(np.mean(errors)),
            "rms_px": float(np.sqrt(np.mean(errors**2))),
            "max_px": float(np.max(errors)),
            "anchors": {
                name: {
                    "error_px": float(error),
                    "confidence": float(confidence),
                    "target_xy": [float(value) for value in target],
                    "projected_xy": [float(value) for value in point],
                }
                for name, error, confidence, target, point in zip(
                    observation.anchor_names,
                    errors,
                    observation.confidence,
                    observation.target_xy,
                    local,
                )
            },
        }
    return result


def _project_all_landmarks(
    vertices: np.ndarray,
    context: NasalBaseOptimizationContext,
    semantic_view: str,
) -> np.ndarray:
    points = np.sum(
        np.asarray(vertices)[context.landmark_triangles]
        * context.landmark_barycentric[:, :, None],
        axis=1,
    )
    view = context.views_by_name[semantic_view]
    camera = (view.R_model_to_camera @ points.T).T + view.t_model_to_camera
    if np.any(camera[:, 2] <= 1e-6):
        raise RuntimeError(f"{semantic_view} full landmark projection is behind camera")
    homogeneous = (view.K @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _fit_similarity_2d(
    source_xy: np.ndarray,
    target_xy: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    source = np.asarray(source_xy, dtype=np.float64)
    target = np.asarray(target_xy, dtype=np.float64)
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    left, singular_values, right_t = np.linalg.svd(covariance)
    rotation = left @ right_t
    if float(np.linalg.det(rotation)) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right_t
        singular_values[-1] *= -1.0
    denominator = float(np.sum(source_zero**2))
    if denominator <= 1e-12:
        raise RuntimeError("protected landmarks do not span a 2-D similarity fit")
    scale = float(np.sum(singular_values) / denominator)
    translation = target_center - scale * (source_center @ rotation)
    return scale, rotation, translation


def _apply_similarity_2d(
    points: np.ndarray,
    scale: float,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> np.ndarray:
    return float(scale) * (np.asarray(points) @ np.asarray(rotation)) + translation


def _registration_diagnostics(
    landmarks_478: Mapping[str, np.ndarray],
    context: NasalBaseOptimizationContext,
) -> dict[str, Any]:
    from src.module2_geometry import _mediapipe_to_68

    # Jaw contour is view-dependent and mouth expression is not a camera cue.
    protected_registration_indices = np.r_[17:31, 36:48]
    result = {}
    for semantic_view in ("front", "subject-left", "subject-right"):
        projected = _project_all_landmarks(
            context.baseline_vertices,
            context,
            semantic_view,
        )
        observed = np.asarray(
            _mediapipe_to_68(landmarks_478[semantic_view]),
            dtype=np.float64,
        )
        scale, rotation, translation = _fit_similarity_2d(
            projected[protected_registration_indices],
            observed[protected_registration_indices],
        )
        registered = _apply_similarity_2d(
            projected,
            scale,
            rotation,
            translation,
        )

        def summary(indices: np.ndarray, values: np.ndarray) -> dict[str, float]:
            errors = np.linalg.norm(values[indices] - observed[indices], axis=1)
            return {
                "mean_px": float(np.mean(errors)),
                "rms_px": float(np.sqrt(np.mean(errors**2))),
                "max_px": float(np.max(errors)),
            }

        result[semantic_view] = {
            "protected_landmark_indices": [
                int(index) for index in protected_registration_indices
            ],
            "transform": {
                "scale": scale,
                "rotation_degrees": float(
                    np.degrees(np.arctan2(rotation[0, 1], rotation[0, 0]))
                ),
                "translation_xy": [float(value) for value in translation],
            },
            "protected_before": summary(
                protected_registration_indices,
                projected,
            ),
            "protected_after": summary(
                protected_registration_indices,
                registered,
            ),
            "nasal_31_35_before": summary(np.arange(31, 36), projected),
            "nasal_31_35_after": summary(np.arange(31, 36), registered),
        }
    return result


def _draw_diagnostics(
    output: Path,
    images: Mapping[str, np.ndarray],
    observations: NasalBaseObservationBundle,
    context: NasalBaseOptimizationContext,
    candidate_vertices: np.ndarray,
) -> Path:
    debug = output / "debug" / "nasal_base_observations"
    debug.mkdir(parents=True, exist_ok=True)
    image_paths = {}
    for semantic_view, observation in observations.by_view.items():
        canvas = np.asarray(images[semantic_view]).copy()
        if canvas.ndim == 2:
            canvas = cv2.cvtColor(canvas, cv2.COLOR_GRAY2RGB)
        baseline_projected = project_nasal_base_landmarks(
            context.baseline_vertices,
            context.landmark_triangles,
            context.landmark_barycentric,
            context.views_by_name[semantic_view],
        )
        candidate_projected = project_nasal_base_landmarks(
            candidate_vertices,
            context.landmark_triangles,
            context.landmark_barycentric,
            context.views_by_name[semantic_view],
        )
        selected = observation.landmark_68_indices - 31
        for source, target, before, after in zip(
            observation.source_xy,
            observation.target_xy,
            baseline_projected[selected],
            candidate_projected[selected],
        ):
            points = (
                (source, (255, 220, 0), 2),
                (target, (255, 0, 210), 3),
                (before, (255, 150, 0), 2),
                (after, (0, 255, 120), 2),
            )
            for point, color, radius in points:
                cv2.circle(
                    canvas,
                    tuple(int(round(value)) for value in point),
                    radius,
                    color,
                    -1,
                    lineType=cv2.LINE_AA,
                )
        path = debug / f"{semantic_view}.png"
        cv2.imwrite(str(path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))
        image_paths[semantic_view] = path

    cards = "\n".join(
        (
            '<section><h2>'
            + html.escape(view)
            + '</h2><img src="'
            + html.escape(path.name, quote=True)
            + '" alt="'
            + html.escape(view, quote=True)
            + ' nasal-base projection overlay"></section>'
        )
        for view, path in image_paths.items()
    )
    report = debug / "index.html"
    report.write_text(
        """<!doctype html>
<html lang="zh-CN"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>A2 鼻基底观测诊断</title>
<style>
body{margin:0;background:#10151b;color:#e7edf3;font:15px Arial,sans-serif}
header{padding:18px 22px;border-bottom:1px solid #35414d}
main{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px;padding:12px}
section{min-width:0}h1,h2{font-size:16px;margin:0 0 10px;letter-spacing:0}
img{display:block;width:100%;height:auto;background:#080b0f}
p{margin:8px 0 0;color:#aab7c4}
@media(max-width:900px){main{grid-template-columns:1fr}}
</style></head><body>
<header><h1>A2 鼻基底观测诊断</h1>
<p>青：478 原始点；紫：局部图像细化；橙：v4；绿：A2。</p></header>
<main>"""
        + cards
        + "</main></body></html>",
        encoding="utf-8",
    )
    return report


def _rename_viewer_labels(path: Path) -> None:
    text = path.read_text(encoding="utf-8")
    text = text.replace(
        "Baseline: protected expression depth v3",
        "Baseline: balanced nasal v4",
    )
    text = text.replace(
        "New: unified multiview nasal shape",
        "Candidate: local nasal-base A2",
    )
    path.write_text(text, encoding="utf-8")


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def run_nasal_base_shape_experiment(
    capture_dir: str | Path,
    source_v4: str | Path,
    output: str | Path,
    *,
    rig_calibration: str | Path,
    expected_geometry_sha256: str = DEFAULT_V4_GEOMETRY_SHA256,
    expected_textured_sha256: str = DEFAULT_V4_TEXTURED_SHA256,
    viewer_template: str | Path | None = None,
) -> Path:
    """Refine only the nasal base and emit a same-texture v4/A2 comparison."""
    captures = Path(capture_dir).resolve()
    source = Path(source_v4).resolve()
    target = Path(output).resolve()
    rig = Path(rig_calibration).resolve()
    if not captures.is_dir():
        raise FileNotFoundError(f"capture directory does not exist: {captures}")
    if not source.is_dir():
        raise FileNotFoundError(f"v4 source output does not exist: {source}")
    if not rig.is_file():
        raise FileNotFoundError(f"rig calibration does not exist: {rig}")
    if target.exists():
        raise FileExistsError(f"refusing to overwrite A2 output: {target}")

    source_geometry = source / "meshes" / "candidate.glb"
    source_textured = source / "meshes" / "candidate_textured.glb"
    geometry_hash = verify_hash_locked_file(
        source_geometry,
        expected_geometry_sha256,
    )
    textured_hash = verify_hash_locked_file(
        source_textured,
        expected_textured_sha256,
    )
    source_report_path = source / "fit_report.json"
    source_report_hash = _sha256_file(source_report_path)
    source_report = json.loads(source_report_path.read_text(encoding="utf-8"))
    if source_report.get("schema") != "balanced-semantic-nasal-experiment-v4":
        raise RuntimeError("source report is not balanced semantic nasal v4")

    target.mkdir(parents=True)
    v4_baseline = _load_v4_low_resolution_baseline(source, source_report)
    nasal_observations_path = source / "nasal_observations.json"
    rig_observations = load_nasal_observation_bundle(nasal_observations_path)
    initial_views = build_model_projection_views(
        rig_observations,
        v4_baseline.front_rotation,
        v4_baseline.front_translation,
    )
    work_images = _load_observation_work_images(
        nasal_observations_path,
        rig_observations,
        rig,
    )
    landmarks = _detect_work_landmarks(work_images)
    from src.module2_geometry import _mediapipe_to_68

    pose_context = RigConsistentPoseContext(
        vertices=v4_baseline.vertices,
        landmark_triangles=v4_baseline.landmark_triangles,
        landmark_barycentric=v4_baseline.landmark_barycentric,
        initial_views=initial_views,
        observed_landmarks_68={
            semantic_view: _mediapipe_to_68(points)
            for semantic_view, points in landmarks.items()
        },
    )
    pose_result = fit_rig_consistent_pose(pose_context)
    if not pose_result.success:
        raise RuntimeError("joint rig-consistent head-pose refinement failed")
    # The shared-pose fit is diagnostic only here. Its protected 68-point
    # semantic mismatch can worsen the already-good frontal nasal alignment.
    # A2 instead removes image-plane nuisance motion in its local objective.
    views = initial_views
    observations = build_nasal_base_observations(
        work_images,
        landmarks,
        refine_to_image=True,
    )
    basis = build_nasal_base_semantic_basis(
        v4_baseline.vertices,
        v4_baseline.faces,
        v4_baseline.landmark_triangles,
        v4_baseline.landmark_barycentric,
        views[0].R_model_to_camera,
    )
    context = NasalBaseOptimizationContext(
        baseline_vertices=v4_baseline.vertices,
        landmark_triangles=v4_baseline.landmark_triangles,
        landmark_barycentric=v4_baseline.landmark_barycentric,
        basis=basis,
        views=views,
        observations=observations,
    )
    result = fit_staged_nasal_base(context)
    if not result.success:
        raise RuntimeError(
            f"A2 nasal-base optimization failed at {result.failure_stage}"
        )

    unchanged_outside_support = bool(
        np.array_equal(
            result.candidate_vertices[~basis.support_mask],
            v4_baseline.vertices[~basis.support_mask],
        )
    )
    unchanged_protected = bool(
        np.array_equal(
            result.candidate_vertices[basis.protected_mask],
            v4_baseline.vertices[basis.protected_mask],
        )
    )
    quality = validate_minimal_nasal_candidate(
        parameters=result.coefficients,
        baseline_vertices=v4_baseline.vertices,
        candidate_vertices=result.candidate_vertices,
        baseline_faces=v4_baseline.faces,
        candidate_faces=v4_baseline.faces,
        baseline_uv_vertices=v4_baseline.uv_vertices,
        candidate_uv_vertices=v4_baseline.uv_vertices,
        baseline_uv_faces=v4_baseline.uv_faces,
        candidate_uv_faces=v4_baseline.uv_faces,
    )
    baseline_metrics = _projection_metrics(v4_baseline.vertices, context)
    candidate_metrics = _projection_metrics(result.candidate_vertices, context)
    registration_diagnostics = _registration_diagnostics(landmarks, context)
    diagnostic = _draw_diagnostics(
        target,
        work_images,
        observations,
        context,
        result.candidate_vertices,
    )
    coefficient_bound = 2.75
    saturated_parameters = [
        name
        for name, value in zip(basis.names, result.coefficients)
        if abs(float(value)) >= coefficient_bound - 1e-6
    ]
    if not unchanged_outside_support:
        quality["issues"].append("vertices_changed_outside_nasal_base_support")
        quality["passed"] = False
    if not unchanged_protected:
        quality["issues"].append("protected_vertices_changed")
        quality["passed"] = False
    if not quality["passed"]:
        _write_json_atomic(
            target / "nasal_base_report.json",
            {
                "schema": "nasal-base-semantic-a2-v1",
                "status": "quality_gate_failed",
                "quality": quality,
                "optimization": result.to_report(),
                "rig_consistent_pose": {
                    **pose_result.to_report(),
                    "applied_to_a2": False,
                    "reason": "A2 uses outer-alar-relative local coordinates",
                },
                "saturated_parameters": saturated_parameters,
                "projection_metrics": {
                    "v4_baseline": baseline_metrics,
                    "a2_candidate": candidate_metrics,
                },
                "registration_diagnostics": registration_diagnostics,
                "observation_diagnostic": str(diagnostic),
            },
        )
        raise RuntimeError(
            "A2 nasal-base candidate failed geometry quality: "
            + ", ".join(quality["issues"])
        )

    artifacts = _export_candidate(
        source_output=source,
        output=target,
        baseline=v4_baseline,
        candidate_vertices=result.candidate_vertices,
    )
    baseline_copy = target / "meshes" / "baseline_v4.glb"
    baseline_copy_hash = _copy_new(source_textured, baseline_copy)
    candidate_geometry = target / "meshes" / "candidate.glb"
    candidate_geometry_hash = _copy_new(
        Path(artifacts["candidate_geometry_glb"]),
        candidate_geometry,
    )
    candidate_textured = target / "meshes" / "candidate_textured.glb"
    candidate_textured_hash = _copy_new(
        Path(artifacts["candidate_textured_glb"]),
        candidate_textured,
    )
    viewer = _write_viewer(
        template=Path(viewer_template).resolve() if viewer_template else None,
        baseline_glb=baseline_copy,
        candidate_glb=candidate_textured,
        output=target / "nasal_base_compare.html",
        dataset_label=f"{captures.name} | nasal base A2",
    )
    _rename_viewer_labels(viewer)
    source_unchanged = {
        "geometry": _sha256_file(source_geometry) == geometry_hash,
        "textured": _sha256_file(source_textured) == textured_hash,
        "report": _sha256_file(source_report_path) == source_report_hash,
    }
    if not all(source_unchanged.values()):
        raise RuntimeError("hash-locked v4 source changed during A2 experiment")

    report = target / "nasal_base_report.json"
    _write_json_atomic(
        report,
        {
            "schema": "nasal-base-semantic-a2-v1",
            "status": "success",
            "dataset": captures.name,
            "texture_scoring_in_objective": False,
            "source_v4": {
                "path": str(source),
                "geometry_sha256": geometry_hash,
                "textured_sha256": textured_hash,
                "report_sha256": source_report_hash,
                "unchanged_after_run": source_unchanged,
            },
            "semantic_model": {
                "parameter_ordering": list(basis.names),
                "unit_scale": float(basis.unit_scale),
                "support_vertex_count": int(np.count_nonzero(basis.support_mask)),
                "protected_vertex_count": int(
                    np.count_nonzero(basis.protected_mask)
                ),
                "outside_support_unchanged": unchanged_outside_support,
                "protected_vertices_unchanged": unchanged_protected,
            },
            "optimization": result.to_report(),
            "rig_consistent_pose": {
                **pose_result.to_report(),
                "applied_to_a2": False,
                "reason": "A2 uses outer-alar-relative local coordinates",
            },
            "saturated_parameters": saturated_parameters,
            "projection_metrics": {
                "v4_baseline": baseline_metrics,
                "a2_candidate": candidate_metrics,
            },
            "registration_diagnostics": registration_diagnostics,
            "quality": quality,
            "artifacts": {
                "baseline_v4_glb": str(baseline_copy),
                "baseline_v4_glb_sha256": baseline_copy_hash,
                "candidate_geometry_glb": str(candidate_geometry),
                "candidate_geometry_glb_sha256": candidate_geometry_hash,
                "candidate_textured_glb": str(candidate_textured),
                "candidate_textured_glb_sha256": candidate_textured_hash,
                "viewer": str(viewer),
                "observation_diagnostic": str(diagnostic),
                "baseline_locked_texture_sha256": artifacts[
                    "baseline_locked_texture_sha256"
                ],
            },
        },
    )
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run A2 local nasal-base semantic refinement."
    )
    parser.add_argument("--capture-dir", type=Path, required=True)
    parser.add_argument("--source-v4", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--rig-calibration", type=Path, required=True)
    parser.add_argument(
        "--expected-geometry-sha256",
        default=DEFAULT_V4_GEOMETRY_SHA256,
    )
    parser.add_argument(
        "--expected-textured-sha256",
        default=DEFAULT_V4_TEXTURED_SHA256,
    )
    parser.add_argument("--viewer-template", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = run_nasal_base_shape_experiment(
        args.capture_dir,
        args.source_v4,
        args.output,
        rig_calibration=args.rig_calibration,
        expected_geometry_sha256=args.expected_geometry_sha256,
        expected_textured_sha256=args.expected_textured_sha256,
        viewer_template=args.viewer_template,
    )
    print(f"A2 nasal-base report: {report}")


if __name__ == "__main__":
    main()

"""Auditable semantic observations for fixed-rig facial profile depth."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import cv2
import numpy as np

from src.cross_view_geometry import Camera
from src.geometry.profile_triangulation import (
    PROFILE_VIEWS,
    ProfileRig,
    TriangulationThresholds,
    project_reference_point,
    triangulate_profile_point,
)
from src.geometry.semantic_epipolar_refinement import (
    SemanticRefinementThresholds,
    build_image_consistent_semantic_observations,
)


SEMANTIC_DEFINITIONS: dict[str, dict[str, tuple[int, ...]]] = {
    "nose_tip": {"mediapipe": (4,), "face_alignment": (30,)},
    "subnasale": {"mediapipe": (2,), "face_alignment": (33,)},
    "upper_lip": {"mediapipe": (0,), "face_alignment": (51,)},
    "lower_lip": {"mediapipe": (17,), "face_alignment": (57,)},
    "mouth_center": {"mediapipe": (13, 14), "face_alignment": (62, 66)},
    "chin": {"mediapipe": (152,), "face_alignment": (8,)},
}


@dataclass(frozen=True)
class ObservationThresholds:
    max_detector_disagreement_ratio: float = 0.04
    max_consensus_correction_px: float = 40.0
    min_detector_confidence: float = 0.20
    min_valid_points: int = 4


def _semantic_points(
    landmarks: Any,
    detector: str,
) -> dict[str, np.ndarray]:
    points = np.asarray(landmarks, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"{detector} landmarks must have shape (N, 2+)")
    if not np.isfinite(points[:, :2]).all():
        raise ValueError(f"{detector} landmarks contain non-finite values")

    result: dict[str, np.ndarray] = {}
    for name, definition in SEMANTIC_DEFINITIONS.items():
        indices = definition[detector]
        if max(indices) >= len(points):
            raise ValueError(
                f"{detector} has {len(points)} landmarks; {name} needs {indices}"
            )
        result[name] = points[list(indices), :2].mean(axis=0)
    return result


def semantic_points_from_mediapipe(landmarks: Any) -> dict[str, np.ndarray]:
    return _semantic_points(landmarks, "mediapipe")


def semantic_points_from_face_alignment(landmarks: Any) -> dict[str, np.ndarray]:
    return _semantic_points(landmarks, "face_alignment")


def undistort_semantic_points(
    points: Mapping[str, Any],
    camera: Camera,
) -> dict[str, np.ndarray]:
    """Undistort original-resolution pixels while retaining the original K frame."""
    names = list(points)
    if not names:
        return {}
    pixels = np.asarray([points[name] for name in names], dtype=np.float64)
    if pixels.shape != (len(names), 2) or not np.isfinite(pixels).all():
        raise ValueError("semantic observations must be finite 2D pixels")
    undistorted = cv2.undistortPoints(
        pixels.reshape(-1, 1, 2),
        camera.K,
        camera.dist,
        P=camera.K,
    ).reshape(-1, 2)
    return {name: undistorted[index] for index, name in enumerate(names)}


def _image_diagonal(image_shape: tuple[int, ...]) -> float:
    if len(image_shape) < 2:
        raise ValueError("image shape must contain height and width")
    height, width = int(image_shape[0]), int(image_shape[1])
    if height <= 0 or width <= 0:
        raise ValueError("image dimensions must be positive")
    return math.hypot(width, height)


def _gradient_response(image: np.ndarray | None, point: np.ndarray) -> float | None:
    if image is None:
        return None
    frame = np.asarray(image)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    elif frame.ndim == 2:
        gray = frame
    else:
        raise ValueError("image must be grayscale or RGB")
    x, y = np.rint(point).astype(np.int64)
    if x < 0 or x >= gray.shape[1] or y < 0 or y >= gray.shape[0]:
        return 0.0
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    magnitude = cv2.magnitude(gx, gy)
    normalizer = float(np.percentile(magnitude, 95.0))
    if normalizer <= 1e-6:
        return 0.0
    y0, y1 = max(0, y - 3), min(gray.shape[0], y + 4)
    x0, x1 = max(0, x - 3), min(gray.shape[1], x + 4)
    return float(np.clip(np.mean(magnitude[y0:y1, x0:x1]) / normalizer, 0.0, 1.0))


def _detector_confidence(disagreement_ratio: float, limit: float) -> float:
    if not np.isfinite(disagreement_ratio) or limit <= 0.0:
        return 0.0
    return float(np.clip(math.exp(-0.5 * (disagreement_ratio / limit) ** 2), 0.0, 1.0))


def _profile_depth_metrics(points: Mapping[str, np.ndarray]) -> dict[str, float]:
    def lead(first: str, second: str) -> float:
        # Front-camera +Z points away from the camera, so a smaller Z is more anterior.
        return float((points[second][2] - points[first][2]) * 1000.0)

    metrics: dict[str, float] = {}
    pairs = (
        ("nose_tip_minus_upper_lip", "nose_tip", "upper_lip"),
        ("nose_tip_minus_lower_lip", "nose_tip", "lower_lip"),
        ("subnasale_minus_mouth", "subnasale", "mouth_center"),
        ("mouth_minus_chin", "mouth_center", "chin"),
    )
    for label, first, second in pairs:
        if first in points and second in points:
            metrics[f"{label}_mm"] = lead(first, second)
    return metrics


def evaluate_semantic_profile_observations(
    mediapipe_by_view: Mapping[str, Mapping[str, Any]],
    face_alignment_by_view: Mapping[str, Mapping[str, Any]],
    rig: ProfileRig,
    image_shapes_by_view: Mapping[str, tuple[int, ...]],
    *,
    images_by_view: Mapping[str, np.ndarray] | None = None,
    observations_undistorted_by_view: Mapping[
        str, Mapping[str, Any]
    ] | None = None,
    image_refinement: Mapping[str, Any] | None = None,
    observation_thresholds: ObservationThresholds | None = None,
    triangulation_thresholds: TriangulationThresholds | None = None,
) -> dict[str, Any]:
    """Evaluate semantic detections without silently turning consensus into truth."""
    limits = observation_thresholds or ObservationThresholds()
    triangle_limits = triangulation_thresholds or TriangulationThresholds()
    detector_details: dict[str, dict[str, Any]] = {}
    undistorted_mp: dict[str, dict[str, np.ndarray]] = {}

    for view in PROFILE_VIEWS:
        if view not in mediapipe_by_view:
            continue
        camera = rig.cameras_by_view[view]
        mp_points = {
            name: np.asarray(point, dtype=np.float64).reshape(2)
            for name, point in mediapipe_by_view[view].items()
        }
        fa_points = {
            name: np.asarray(point, dtype=np.float64).reshape(2)
            for name, point in face_alignment_by_view.get(view, {}).items()
        }
        undistorted_mp[view] = undistort_semantic_points(mp_points, camera)
        diagonal = _image_diagonal(image_shapes_by_view[view])
        detector_details[view] = {}
        for name, point in mp_points.items():
            fa_point = fa_points.get(name)
            disagreement = (
                float(np.linalg.norm(point - fa_point))
                if fa_point is not None else float("inf")
            )
            disagreement_ratio = disagreement / diagonal
            detector_details[view][name] = {
                "mediapipe_px": point.astype(float).tolist(),
                "face_alignment_px": (
                    None if fa_point is None else fa_point.astype(float).tolist()
                ),
                "detector_disagreement_px": disagreement,
                "detector_disagreement_ratio": disagreement_ratio,
                "detector_confidence": _detector_confidence(
                    disagreement_ratio,
                    limits.max_detector_disagreement_ratio,
                ),
                "boundary_response": _gradient_response(
                    None if images_by_view is None else images_by_view.get(view),
                    point,
                ),
            }

    if observations_undistorted_by_view is not None:
        undistorted_mp = {
            view: {
                name: np.asarray(point, dtype=np.float64).reshape(2)
                for name, point in points.items()
            }
            for view, points in observations_undistorted_by_view.items()
        }
    semantic_names = sorted(
        {
            name
            for points in undistorted_mp.values()
            for name in points
            if sum(name in candidate for candidate in undistorted_mp.values()) >= 2
        }
    )
    point_reports: dict[str, dict[str, Any]] = {}
    accepted_points: dict[str, np.ndarray] = {}
    for name in semantic_names:
        observations = {
            view: undistorted_mp[view][name]
            for view in PROFILE_VIEWS
            if view in undistorted_mp and name in undistorted_mp[view]
        }
        if len(observations) < 2:
            continue
        triangulation = triangulate_profile_point(
            observations,
            rig,
            thresholds=triangle_limits,
        )
        point = np.asarray(triangulation["point_reference_m"], dtype=np.float64)
        consensus = {
            view: project_reference_point(point, view, rig)
            for view in observations
        }
        corrections = {
            view: float(np.linalg.norm(consensus[view] - observations[view]))
            for view in observations
        }
        correction_values = np.asarray(list(corrections.values()), dtype=np.float64)
        correction_p90 = float(np.percentile(correction_values, 90.0))
        confidence_views = (
            ("front",)
            if image_refinement is not None and "front" in observations
            else tuple(observations)
        )
        detector_confidences = [
            float(detector_details[view][name]["detector_confidence"])
            for view in confidence_views
            if view in detector_details and name in detector_details[view]
        ]
        mean_detector_confidence = float(np.mean(detector_confidences))
        local_consensus = correction_p90 <= limits.max_consensus_correction_px
        passed = bool(
            triangulation["passed"]
            and local_consensus
            and mean_detector_confidence >= limits.min_detector_confidence
        )
        issues = list(triangulation["issues"])
        if not local_consensus:
            issues.append("semantic_correction_exceeded")
        if mean_detector_confidence < limits.min_detector_confidence:
            issues.append("detector_disagreement_exceeded")
        issues = list(dict.fromkeys(issues))
        if passed:
            accepted_points[name] = point
        point_reports[name] = {
            "passed": passed,
            "issues": issues,
            "triangulation": triangulation,
            "epipolar_consensus": {
                "reprojected_px": {
                    view: value.astype(float).tolist()
                    for view, value in consensus.items()
                },
                "correction_px": corrections,
                "correction_p90_px": correction_p90,
                "max_allowed_correction_px": float(
                    limits.max_consensus_correction_px
                ),
                "within_local_roi": local_consensus,
            },
            "mean_detector_confidence": mean_detector_confidence,
            "observation_view_count": len(observations),
            "three_view_validated": len(observations) == 3,
            "observation_sources": {
                view: (
                    "mediapipe_front_anchor"
                    if image_refinement is not None and view == "front"
                    else str(image_refinement.get("method", "image_refinement"))
                    if image_refinement is not None
                    else "mediapipe_semantic_index"
                )
                for view in observations
            },
            "image_refinement": (
                None
                if image_refinement is None
                else image_refinement.get("points", {}).get(name, {})
            ),
        }

    valid_names = sorted(accepted_points)
    has_lip = "upper_lip" in accepted_points or "lower_lip" in accepted_points
    missing_required = [
        name for name in ("nose_tip", "chin") if name not in accepted_points
    ]
    if not has_lip:
        missing_required.append("upper_or_lower_lip")
    quality_issues = []
    quality_warnings = []
    if len(valid_names) < limits.min_valid_points:
        quality_issues.append("insufficient_valid_semantic_points")
    if missing_required:
        quality_issues.append("required_semantic_points_missing")
    two_view_only = sorted(
        name
        for name in accepted_points
        if not point_reports[name]["three_view_validated"]
    )
    if two_view_only:
        quality_warnings.append("accepted_points_without_three_view_validation")

    return {
        "observation_version": 2 if image_refinement is not None else 1,
        "audit_only": True,
        "semantic_definitions": {
            name: {
                detector: list(indices)
                for detector, indices in definition.items()
            }
            for name, definition in SEMANTIC_DEFINITIONS.items()
        },
        "detectors": detector_details,
        "undistorted_mediapipe_px": {
            view: {
                name: point.astype(float).tolist()
                for name, point in points.items()
            }
            for view, points in undistorted_mp.items()
        },
        "points": point_reports,
        "accepted_points_reference_m": {
            name: point.astype(float).tolist()
            for name, point in accepted_points.items()
        },
        "observed_profile_depth": _profile_depth_metrics(accepted_points),
        "image_refinement": image_refinement,
        "quality_gate": {
            "passed": not quality_issues,
            "issues": quality_issues,
            "valid_points": valid_names,
            "valid_point_count": len(valid_names),
            "minimum_valid_points": int(limits.min_valid_points),
            "missing_required": missing_required,
            "warnings": quality_warnings,
            "two_view_only_points": two_view_only,
        },
        "thresholds": {
            "max_detector_disagreement_ratio": float(
                limits.max_detector_disagreement_ratio
            ),
            "max_consensus_correction_px": float(
                limits.max_consensus_correction_px
            ),
            "min_detector_confidence": float(limits.min_detector_confidence),
            "min_valid_points": int(limits.min_valid_points),
        },
        "rig": {
            "calibration_path": rig.calibration_path,
            "reference_view": rig.reference_view,
            "units": rig.units,
            "stereo_rms_px": dict(rig.stereo_rms_px),
        },
    }


def collect_profile_observations(
    mediapipe_landmarks_by_view: Mapping[str, Any],
    face_alignment_landmarks_by_view: Mapping[str, Any],
    rig: ProfileRig,
    image_shapes_by_view: Mapping[str, tuple[int, ...]],
    dense_matches_by_view: Mapping[str, Mapping[str, Any]] | None = None,
    semantic_refinement_thresholds: SemanticRefinementThresholds | None = None,
    **kwargs: Any,
) -> dict[str, Any]:
    mediapipe_semantic = {
        view: semantic_points_from_mediapipe(points)
        for view, points in mediapipe_landmarks_by_view.items()
        if points is not None
    }
    face_alignment_semantic = {
        view: semantic_points_from_face_alignment(points)
        for view, points in face_alignment_landmarks_by_view.items()
        if points is not None
    }
    refinement_kwargs: dict[str, Any] = {}
    if dense_matches_by_view is not None:
        observations, refinement = build_image_consistent_semantic_observations(
            mediapipe_semantic,
            dense_matches_by_view,
            rig,
            thresholds=semantic_refinement_thresholds,
        )
        refinement_kwargs = {
            "observations_undistorted_by_view": observations,
            "image_refinement": refinement,
        }
    return evaluate_semantic_profile_observations(
        mediapipe_semantic,
        face_alignment_semantic,
        rig,
        image_shapes_by_view,
        **refinement_kwargs,
        **kwargs,
    )


def write_profile_points_ply(report: Mapping[str, Any], path: str | Path) -> None:
    """Write every finite triangulated point; red vertices failed their gates."""
    vertices: list[tuple[float, float, float, int, int, int]] = []
    for point_report in report.get("points", {}).values():
        point = np.asarray(
            point_report["triangulation"]["point_reference_m"], dtype=np.float64
        )
        if not np.isfinite(point).all():
            continue
        color = (62, 207, 142) if point_report.get("passed") else (239, 107, 107)
        vertices.append((*point.astype(float), *color))
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [
        "ply",
        "format ascii 1.0",
        f"element vertex {len(vertices)}",
        "property float x",
        "property float y",
        "property float z",
        "property uchar red",
        "property uchar green",
        "property uchar blue",
        "end_header",
    ]
    lines.extend(
        f"{x:.9f} {y:.9f} {z:.9f} {r} {g} {b}"
        for x, y, z, r, g, b in vertices
    )
    target.write_text("\n".join(lines) + "\n", encoding="ascii")

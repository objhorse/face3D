"""Fixed-rig triangulation for auditable semantic face profile points."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.cross_view_geometry import Camera, relative_camera_transform


PROFILE_VIEWS = ("left", "front", "right")


@dataclass(frozen=True)
class TriangulationThresholds:
    max_reprojection_px: float = 5.0
    min_ray_angle_deg: float = 3.0
    max_pair_delta_m: float = 0.005
    min_depth_m: float = 0.12
    max_depth_m: float = 1.50


@dataclass(frozen=True)
class ProfileRig:
    cameras_by_view: Mapping[str, Camera]
    reference_view: str
    units: str
    calibration_path: str
    stereo_rms_px: Mapping[str, float]


def _rotation_is_valid(rotation: np.ndarray, tolerance: float = 1e-4) -> bool:
    matrix = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return bool(
        np.isfinite(matrix).all()
        and np.allclose(matrix @ matrix.T, np.eye(3), atol=tolerance)
        and abs(float(np.linalg.det(matrix)) - 1.0) <= tolerance
    )


def load_profile_rig(
    path: str | Path,
    *,
    max_stereo_rms_px: float = 10.0,
    expected_camera_by_view: Mapping[str, str] | None = None,
) -> ProfileRig:
    """Load a metric three-camera rig and reject stale or mislabeled calibration."""
    calibration_path = Path(path).resolve()
    payload = json.loads(calibration_path.read_text(encoding="utf-8"))
    return profile_rig_from_payload(
        payload,
        calibration_path=str(calibration_path),
        max_stereo_rms_px=max_stereo_rms_px,
        expected_camera_by_view=expected_camera_by_view,
    )


def profile_rig_from_payload(
    payload: Mapping[str, Any],
    *,
    calibration_path: str = "<memory>",
    max_stereo_rms_px: float = 10.0,
    expected_camera_by_view: Mapping[str, str] | None = None,
) -> ProfileRig:
    """Validate an already parsed calibration payload."""
    units = str(payload.get("units", "")).strip().lower()
    if units not in {"meter", "meters", "m"}:
        raise ValueError(f"profile rig must use meters, found {units!r}")

    cameras_by_name: dict[str, Camera] = {}
    for name, item in payload.get("cameras", {}).items():
        cameras_by_name[str(name)] = Camera(
            name=str(name),
            view=str(item["view"]),
            image_size=tuple(int(value) for value in item["image_size"]),
            K=np.asarray(item["K"], dtype=np.float64),
            dist=np.asarray(item["dist_coeffs"], dtype=np.float64).reshape(-1),
            R_rig_to_camera=np.asarray(
                item["rig_to_camera"]["R"], dtype=np.float64
            ),
            t_rig_to_camera=np.asarray(
                item["rig_to_camera"]["t"], dtype=np.float64
            ).reshape(3),
        )
    cameras_by_view: dict[str, Camera] = {}
    for camera in cameras_by_name.values():
        if camera.view in cameras_by_view:
            raise ValueError(f"duplicate camera view in calibration: {camera.view}")
        cameras_by_view[camera.view] = camera
    missing = [view for view in PROFILE_VIEWS if view not in cameras_by_view]
    if missing:
        raise ValueError(f"profile rig is missing views: {', '.join(missing)}")

    expected = expected_camera_by_view or {
        "left": "camera1",
        "front": "camera2",
        "right": "camera3",
    }
    for view, expected_name in expected.items():
        actual = cameras_by_view.get(view)
        if actual is None or actual.name != expected_name:
            actual_name = None if actual is None else actual.name
            raise ValueError(
                f"camera naming mismatch for {view}: expected {expected_name}, found {actual_name}"
            )

    reference_camera_name = str(payload.get("reference_camera", ""))
    if reference_camera_name not in cameras_by_name:
        raise ValueError(f"reference camera is missing: {reference_camera_name}")
    reference_view = cameras_by_name[reference_camera_name].view
    if reference_view != "front":
        raise ValueError(f"profile rig reference must be front, found {reference_view}")

    camera_payload = payload.get("cameras", {})
    stereo_rms: dict[str, float] = {}
    for view in PROFILE_VIEWS:
        camera = cameras_by_view[view]
        if not _rotation_is_valid(camera.R_rig_to_camera):
            raise ValueError(f"invalid rig rotation for {camera.name}/{view}")
        if not np.isfinite(camera.t_rig_to_camera).all():
            raise ValueError(f"invalid rig translation for {camera.name}/{view}")
        raw_rms = camera_payload.get(camera.name, {}).get("stereo_rms")
        if raw_rms is None:
            raise ValueError(f"stereo RMS is missing for {camera.name}/{view}")
        rms = float(raw_rms)
        if not np.isfinite(rms) or rms < 0.0:
            raise ValueError(f"invalid stereo RMS for {camera.name}/{view}: {rms}")
        stereo_rms[view] = rms
        if view != reference_view and rms > float(max_stereo_rms_px):
            raise ValueError(
                f"stereo RMS {rms:.3f}px exceeds {max_stereo_rms_px:.3f}px "
                f"for {camera.name}/{view}"
            )

    return ProfileRig(
        cameras_by_view={view: cameras_by_view[view] for view in PROFILE_VIEWS},
        reference_view=reference_view,
        units="meters",
        calibration_path=str(calibration_path),
        stereo_rms_px=stereo_rms,
    )


def _relative_transform(rig: ProfileRig, view: str) -> tuple[np.ndarray, np.ndarray]:
    if view not in rig.cameras_by_view:
        raise KeyError(f"view is missing from profile rig: {view}")
    reference = rig.cameras_by_view[rig.reference_view]
    camera = rig.cameras_by_view[view]
    return relative_camera_transform(reference, camera)


def _projection_matrix(
    rig: ProfileRig,
    view: str,
    intrinsics_by_view: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    rotation, translation = _relative_transform(rig, view)
    intrinsics = (
        rig.cameras_by_view[view].K
        if intrinsics_by_view is None
        else intrinsics_by_view[view]
    )
    K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(K).all() or abs(float(np.linalg.det(K))) <= 1e-12:
        raise ValueError(f"invalid intrinsics for {view}")
    return K @ np.column_stack([rotation, translation.reshape(3, 1)])


def project_reference_point(
    point_reference: Any,
    view: str,
    rig: ProfileRig,
    intrinsics_by_view: Mapping[str, np.ndarray] | None = None,
) -> np.ndarray:
    point = np.asarray(point_reference, dtype=np.float64).reshape(3)
    projection = _projection_matrix(rig, view, intrinsics_by_view)
    homogeneous = projection @ np.append(point, 1.0)
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[2])) <= 1e-12:
        return np.full(2, np.nan, dtype=np.float64)
    return homogeneous[:2] / homogeneous[2]


def _triangulate_dlt(
    observations: Mapping[str, np.ndarray],
    rig: ProfileRig,
    intrinsics_by_view: Mapping[str, np.ndarray] | None,
    weights_by_view: Mapping[str, float] | None,
) -> np.ndarray:
    rows = []
    for view in sorted(observations):
        point = np.asarray(observations[view], dtype=np.float64).reshape(2)
        projection = _projection_matrix(rig, view, intrinsics_by_view)
        weight = 1.0 if weights_by_view is None else float(weights_by_view.get(view, 1.0))
        if not np.isfinite(point).all():
            raise ValueError(f"non-finite observation for {view}")
        if not np.isfinite(weight) or weight <= 0.0:
            raise ValueError(f"invalid triangulation weight for {view}: {weight}")
        scale = math.sqrt(weight)
        rows.append((point[0] * projection[2] - projection[0]) * scale)
        rows.append((point[1] * projection[2] - projection[1]) * scale)
    matrix = np.asarray(rows, dtype=np.float64)
    _u, _s, vh = np.linalg.svd(matrix, full_matrices=False)
    homogeneous = vh[-1]
    if not np.isfinite(homogeneous).all() or abs(float(homogeneous[3])) <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return homogeneous[:3] / homogeneous[3]


def _camera_center_reference(rig: ProfileRig, view: str) -> np.ndarray:
    rotation, translation = _relative_transform(rig, view)
    return -rotation.T @ translation


def _ray_angle(point: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
    ray_first = point - first
    ray_second = point - second
    denominator = float(np.linalg.norm(ray_first) * np.linalg.norm(ray_second))
    if denominator <= 1e-12:
        return 0.0
    cosine = float(np.dot(ray_first, ray_second) / denominator)
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def triangulate_profile_point(
    observations: Mapping[str, Any],
    rig: ProfileRig,
    *,
    intrinsics_by_view: Mapping[str, np.ndarray] | None = None,
    weights_by_view: Mapping[str, float] | None = None,
    thresholds: TriangulationThresholds | None = None,
) -> dict[str, Any]:
    """Triangulate one semantic point and return all acceptance evidence."""
    limits = thresholds or TriangulationThresholds()
    pixels = {
        str(view): np.asarray(point, dtype=np.float64).reshape(2)
        for view, point in observations.items()
    }
    if len(pixels) < 2:
        raise ValueError("profile triangulation requires at least two views")
    unknown = sorted(set(pixels) - set(rig.cameras_by_view))
    if unknown:
        raise KeyError(f"unknown profile views: {', '.join(unknown)}")

    point = _triangulate_dlt(pixels, rig, intrinsics_by_view, weights_by_view)
    issues: list[str] = []
    depths: dict[str, float] = {}
    reprojection: dict[str, float] = {}
    if not np.isfinite(point).all():
        issues.append("negative_or_invalid_depth")
    else:
        for view, target in pixels.items():
            rotation, translation = _relative_transform(rig, view)
            depth = float((rotation @ point + translation)[2])
            depths[view] = depth
            projected = project_reference_point(
                point, view, rig, intrinsics_by_view
            )
            reprojection[view] = float(np.linalg.norm(projected - target))
        if any(
            not np.isfinite(depth)
            or depth < float(limits.min_depth_m)
            or depth > float(limits.max_depth_m)
            for depth in depths.values()
        ):
            issues.append("negative_or_invalid_depth")

    reprojection_values = np.asarray(list(reprojection.values()), dtype=np.float64)
    reprojection_p90 = (
        float(np.percentile(reprojection_values, 90.0))
        if len(reprojection_values) else float("inf")
    )
    if reprojection_p90 > float(limits.max_reprojection_px):
        issues.append("reprojection_error_exceeded")

    ray_angles: dict[str, float] = {}
    if np.isfinite(point).all():
        centers = {
            view: _camera_center_reference(rig, view)
            for view in pixels
        }
        for first, second in combinations(sorted(centers), 2):
            ray_angles[f"{first}_{second}"] = _ray_angle(
                point, centers[first], centers[second]
            )
    min_ray_angle = min(ray_angles.values(), default=0.0)
    if min_ray_angle < float(limits.min_ray_angle_deg):
        issues.append("ray_angle_too_small")

    pair_points: dict[str, list[float]] = {}
    pair_valid = True
    for first, second in combinations(sorted(pixels), 2):
        pair_point = _triangulate_dlt(
            {first: pixels[first], second: pixels[second]},
            rig,
            intrinsics_by_view,
            weights_by_view,
        )
        pair_points[f"{first}_{second}"] = pair_point.astype(float).tolist()
        if not np.isfinite(pair_point).all():
            pair_valid = False
            continue
        for view in (first, second):
            rotation, translation = _relative_transform(rig, view)
            if float((rotation @ pair_point + translation)[2]) <= 0.0:
                pair_valid = False
    pair_delta = 0.0
    finite_pairs = [
        np.asarray(value, dtype=np.float64)
        for value in pair_points.values()
        if np.isfinite(value).all()
    ]
    if len(finite_pairs) >= 2:
        pair_delta = max(
            float(np.linalg.norm(a - b))
            for a, b in combinations(finite_pairs, 2)
        )
    if not pair_valid:
        issues.append("negative_or_invalid_depth")
    if pair_delta > float(limits.max_pair_delta_m):
        issues.append("pairwise_3d_inconsistency")

    unique_issues = list(dict.fromkeys(issues))
    return {
        "passed": not unique_issues,
        "issues": unique_issues,
        "point_reference_m": point.astype(float).tolist(),
        "observations_px": {
            view: target.astype(float).tolist()
            for view, target in pixels.items()
        },
        "depths_m": depths,
        "reprojection_errors_px": reprojection,
        "reprojection_p90_px": reprojection_p90,
        "ray_angles_deg": ray_angles,
        "min_ray_angle_deg": float(min_ray_angle),
        "pair_points_reference_m": pair_points,
        "max_pair_delta_m": float(pair_delta),
        "thresholds": {
            "max_reprojection_px": float(limits.max_reprojection_px),
            "min_ray_angle_deg": float(limits.min_ray_angle_deg),
            "max_pair_delta_m": float(limits.max_pair_delta_m),
            "min_depth_m": float(limits.min_depth_m),
            "max_depth_m": float(limits.max_depth_m),
        },
        "rig": {
            "reference_view": rig.reference_view,
            "units": rig.units,
            "calibration_path": rig.calibration_path,
            "stereo_rms_px": dict(rig.stereo_rms_px),
        },
    }

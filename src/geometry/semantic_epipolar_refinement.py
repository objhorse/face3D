"""Image-evidence refinement for cross-view semantic face observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    relative_camera_transform,
    scale_intrinsics,
    triangulate_correspondences,
)
from src.geometry.profile_triangulation import (
    ProfileRig,
    project_reference_point,
)


@dataclass(frozen=True)
class SemanticRefinementThresholds:
    local_radius_px: float = 80.0
    side_prior_radius_px: float = 120.0
    max_local_matches: int = 120
    min_local_matches: int = 6
    min_inlier_matches: int = 12
    min_inlier_ratio: float = 0.35
    max_affine_residual_p90_px: float = 4.0
    max_anchor_support_distance_px: float = 40.0
    max_detector_shift_px: float = 90.0
    max_epipolar_correction_px: float = 5.0
    ransac_threshold_px: float = 3.0
    depth_search_radii_px: tuple[float, ...] = (48.0, 64.0, 80.0, 100.0)
    min_depth_surface_inliers: int = 8
    max_depth_surface_residual_p90_m: float = 0.005
    max_dense_epipolar_error_px: float = 5.0
    max_cross_side_depth_delta_m: float = 0.008
    min_face_depth_m: float = 0.12
    max_face_depth_m: float = 1.50


def _scale_point(
    point: Any,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64).reshape(2)
    return value * np.asarray(
        (
            target_size[0] / float(source_size[0]),
            target_size[1] / float(source_size[1]),
        ),
        dtype=np.float64,
    )


def _unscale_point(
    point: Any,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    value = np.asarray(point, dtype=np.float64).reshape(2)
    return value * np.asarray(
        (
            source_size[0] / float(target_size[0]),
            source_size[1] / float(target_size[1]),
        ),
        dtype=np.float64,
    )


def _undistort_work_point(
    point: np.ndarray,
    camera: Camera,
    work_size: tuple[int, int],
) -> np.ndarray:
    intrinsics = scale_intrinsics(camera.K, camera.image_size, work_size)
    return cv2.undistortPoints(
        np.asarray(point, dtype=np.float64).reshape(1, 1, 2),
        intrinsics,
        camera.dist,
        P=intrinsics,
    ).reshape(2)


def _fundamental_matrix(
    camera_a: Camera,
    camera_b: Camera,
    work_size: tuple[int, int],
) -> np.ndarray:
    intrinsics_a = scale_intrinsics(camera_a.K, camera_a.image_size, work_size)
    intrinsics_b = scale_intrinsics(camera_b.K, camera_b.image_size, work_size)
    rotation, translation = relative_camera_transform(camera_a, camera_b)
    tx, ty, tz = np.asarray(translation, dtype=np.float64).reshape(3)
    skew = np.array(
        ((0.0, -tz, ty), (tz, 0.0, -tx), (-ty, tx, 0.0)),
        dtype=np.float64,
    )
    essential = skew @ rotation
    return np.linalg.inv(intrinsics_b).T @ essential @ np.linalg.inv(intrinsics_a)


def _project_to_line(point: np.ndarray, line: np.ndarray) -> tuple[np.ndarray, float]:
    a, b, c = np.asarray(line, dtype=np.float64).reshape(3)
    denominator = float(a * a + b * b)
    if denominator <= 1e-12:
        return np.full(2, np.nan, dtype=np.float64), float("inf")
    x, y = np.asarray(point, dtype=np.float64).reshape(2)
    signed = float((a * x + b * y + c) / denominator)
    projected = np.array((x - a * signed, y - b * signed), dtype=np.float64)
    return projected, float(np.linalg.norm(projected - point))


def refine_semantic_pair(
    anchor_front_px: Any,
    side_detector_px: Any,
    points_front_work: Any,
    points_side_work: Any,
    confidence: Any,
    front_camera: Camera,
    side_camera: Camera,
    *,
    work_size: tuple[int, int] = (640, 480),
    thresholds: SemanticRefinementThresholds | None = None,
) -> dict[str, Any]:
    """Map one front semantic pixel to a side image using nearby LoFTR evidence."""
    limits = thresholds or SemanticRefinementThresholds()
    front_matches = np.asarray(points_front_work, dtype=np.float64).reshape(-1, 2)
    side_matches = np.asarray(points_side_work, dtype=np.float64).reshape(-1, 2)
    weights = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not (
        len(front_matches) == len(side_matches) == len(weights)
        and np.isfinite(front_matches).all()
        and np.isfinite(side_matches).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("dense match arrays must be finite and have equal lengths")

    anchor_work = _scale_point(
        anchor_front_px, front_camera.image_size, work_size
    )
    prior_work = _scale_point(
        side_detector_px, side_camera.image_size, work_size
    )
    source_distance = np.linalg.norm(front_matches - anchor_work, axis=1)
    prior_distance = np.linalg.norm(side_matches - prior_work, axis=1)
    selected = (
        (source_distance <= float(limits.local_radius_px))
        & (prior_distance <= float(limits.side_prior_radius_px))
        & (weights > 0.0)
    )
    indices = np.flatnonzero(selected)
    if len(indices) > int(limits.max_local_matches):
        rank = np.lexsort((-weights[indices], source_distance[indices]))
        indices = indices[rank[: int(limits.max_local_matches)]]

    issues: list[str] = []
    if len(indices) < int(limits.min_local_matches):
        issues.append("insufficient_local_image_matches")
    nearest_support = (
        float(np.min(source_distance[indices])) if len(indices) else float("inf")
    )
    if nearest_support > float(limits.max_anchor_support_distance_px):
        issues.append("semantic_anchor_not_locally_supported")

    affine = None
    inlier_mask = np.zeros(len(indices), dtype=bool)
    if len(indices) >= int(limits.min_local_matches):
        cv2.setRNGSeed(0)
        affine, raw_inliers = cv2.estimateAffine2D(
            front_matches[indices],
            side_matches[indices],
            method=cv2.RANSAC,
            ransacReprojThreshold=float(limits.ransac_threshold_px),
            maxIters=5000,
            confidence=0.995,
            refineIters=20,
        )
        if affine is not None and raw_inliers is not None:
            inlier_mask = raw_inliers.reshape(-1).astype(bool)
    if affine is None or not np.isfinite(affine).all():
        issues.append("local_affine_fit_failed")
        predicted_side = np.full(2, np.nan, dtype=np.float64)
        residuals = np.empty(0, dtype=np.float64)
    else:
        predicted_side = (
            affine[:, :2] @ anchor_work + affine[:, 2]
        ).astype(np.float64)
        predicted_matches = (
            front_matches[indices] @ affine[:, :2].T + affine[:, 2]
        )
        residuals = np.linalg.norm(
            predicted_matches[inlier_mask] - side_matches[indices][inlier_mask],
            axis=1,
        )

    inlier_count = int(np.count_nonzero(inlier_mask))
    inlier_ratio = inlier_count / float(max(1, len(indices)))
    residual_p50 = (
        float(np.percentile(residuals, 50.0)) if len(residuals) else float("inf")
    )
    residual_p90 = (
        float(np.percentile(residuals, 90.0)) if len(residuals) else float("inf")
    )
    if inlier_count < int(limits.min_inlier_matches):
        issues.append("insufficient_local_affine_inliers")
    if inlier_ratio < float(limits.min_inlier_ratio):
        issues.append("local_affine_inlier_ratio_low")
    if residual_p90 > float(limits.max_affine_residual_p90_px):
        issues.append("local_affine_residual_exceeded")

    detector_shift = (
        float(np.linalg.norm(predicted_side - prior_work))
        if np.isfinite(predicted_side).all()
        else float("inf")
    )
    if detector_shift > float(limits.max_detector_shift_px):
        issues.append("refined_point_left_semantic_roi")

    anchor_undistorted = _undistort_work_point(
        anchor_work, front_camera, work_size
    )
    predicted_undistorted = (
        _undistort_work_point(predicted_side, side_camera, work_size)
        if np.isfinite(predicted_side).all()
        else np.full(2, np.nan, dtype=np.float64)
    )
    fundamental = _fundamental_matrix(front_camera, side_camera, work_size)
    epipolar_line = fundamental @ np.append(anchor_undistorted, 1.0)
    corrected_side, epipolar_correction = _project_to_line(
        predicted_undistorted, epipolar_line
    )
    if epipolar_correction > float(limits.max_epipolar_correction_px):
        issues.append("epipolar_correction_exceeded")
    if not np.isfinite(corrected_side).all():
        issues.append("refined_point_non_finite")

    issues = list(dict.fromkeys(issues))
    corrected_original = _unscale_point(
        corrected_side, side_camera.image_size, work_size
    )
    return {
        "passed": not issues,
        "issues": issues,
        "source": "front_semantic_local_affine_loftr",
        "front_anchor_work_px": anchor_work.astype(float).tolist(),
        "side_detector_work_px": prior_work.astype(float).tolist(),
        "predicted_side_work_px": predicted_side.astype(float).tolist(),
        "corrected_side_undistorted_work_px": corrected_side.astype(float).tolist(),
        "corrected_side_undistorted_px": corrected_original.astype(float).tolist(),
        "candidate_matches": int(len(indices)),
        "inlier_matches": inlier_count,
        "inlier_ratio": float(inlier_ratio),
        "nearest_anchor_support_px": nearest_support,
        "affine_residual_p50_px": residual_p50,
        "affine_residual_p90_px": residual_p90,
        "detector_to_prediction_px": detector_shift,
        "epipolar_correction_px": epipolar_correction,
    }


def _dense_depth_samples(
    matches: Mapping[str, Any],
    front_camera: Camera,
    side_camera: Camera,
    *,
    work_size: tuple[int, int],
    thresholds: SemanticRefinementThresholds,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_front = np.asarray(
        matches["points_front_work"], dtype=np.float64
    ).reshape(-1, 2)
    points_side = np.asarray(
        matches["points_side_work"], dtype=np.float64
    ).reshape(-1, 2)
    confidence = np.asarray(matches["confidence"], dtype=np.float64).reshape(-1)
    if not (
        len(points_front) == len(points_side) == len(confidence)
        and np.isfinite(points_front).all()
        and np.isfinite(points_side).all()
        and np.isfinite(confidence).all()
    ):
        raise ValueError("dense depth matches must be finite and aligned")
    intrinsics_front = scale_intrinsics(
        front_camera.K, front_camera.image_size, work_size
    )
    intrinsics_side = scale_intrinsics(
        side_camera.K, side_camera.image_size, work_size
    )
    undistorted_front = cv2.undistortPoints(
        points_front.reshape(-1, 1, 2),
        intrinsics_front,
        front_camera.dist,
        P=intrinsics_front,
    ).reshape(-1, 2)
    undistorted_side = cv2.undistortPoints(
        points_side.reshape(-1, 1, 2),
        intrinsics_side,
        side_camera.dist,
        P=intrinsics_side,
    ).reshape(-1, 2)
    rotation, translation = relative_camera_transform(
        front_camera, side_camera
    )
    xyz, positive = triangulate_correspondences(
        undistorted_front,
        undistorted_side,
        intrinsics_front,
        intrinsics_side,
        rotation,
        translation,
    )
    fundamental = _fundamental_matrix(front_camera, side_camera, work_size)
    homogeneous_front = np.column_stack(
        (undistorted_front, np.ones(len(undistorted_front)))
    )
    lines = homogeneous_front @ fundamental.T
    numerator = np.abs(
        np.sum(
            lines
            * np.column_stack(
                (undistorted_side, np.ones(len(undistorted_side)))
            ),
            axis=1,
        )
    )
    denominator = np.linalg.norm(lines[:, :2], axis=1)
    epipolar_error = numerator / np.maximum(denominator, 1e-12)
    valid = (
        positive
        & np.isfinite(xyz[:, 2])
        & (xyz[:, 2] >= float(thresholds.min_face_depth_m))
        & (xyz[:, 2] <= float(thresholds.max_face_depth_m))
        & (epipolar_error <= float(thresholds.max_dense_epipolar_error_px))
    )
    return points_front[valid], xyz[valid, 2], confidence[valid]


def estimate_local_depth_surface(
    anchor_work_px: Any,
    points_front_work: Any,
    depth_m: Any,
    confidence: Any,
    *,
    thresholds: SemanticRefinementThresholds | None = None,
) -> dict[str, Any]:
    """Fit the smallest supported robust local depth plane at one front pixel."""
    limits = thresholds or SemanticRefinementThresholds()
    anchor = np.asarray(anchor_work_px, dtype=np.float64).reshape(2)
    points = np.asarray(points_front_work, dtype=np.float64).reshape(-1, 2)
    depth = np.asarray(depth_m, dtype=np.float64).reshape(-1)
    weights = np.asarray(confidence, dtype=np.float64).reshape(-1)
    if not (
        len(points) == len(depth) == len(weights)
        and np.isfinite(points).all()
        and np.isfinite(depth).all()
        and np.isfinite(weights).all()
    ):
        raise ValueError("local depth samples must be finite and aligned")
    distance = np.linalg.norm(points - anchor, axis=1)
    best: dict[str, Any] | None = None
    for radius in limits.depth_search_radii_px:
        selected = distance <= float(radius)
        local_points = points[selected]
        local_depth = depth[selected]
        local_confidence = weights[selected]
        local_distance = distance[selected]
        if len(local_points) < int(limits.min_depth_surface_inliers):
            continue
        design = np.column_stack(
            (local_points - anchor, np.ones(len(local_points)))
        )
        spatial_weight = np.exp(
            -0.5 * (local_distance / max(float(radius) * 0.55, 1e-6)) ** 2
        )
        fit_weight = np.maximum(local_confidence, 1e-3) * spatial_weight
        active = np.ones(len(local_points), dtype=bool)
        beta = np.full(3, np.nan, dtype=np.float64)
        residual = np.full(len(local_points), np.inf, dtype=np.float64)
        for _iteration in range(5):
            if int(active.sum()) < int(limits.min_depth_surface_inliers):
                break
            sqrt_weight = np.sqrt(fit_weight[active])
            matrix = design[active] * sqrt_weight[:, None]
            target = local_depth[active] * sqrt_weight
            beta = np.linalg.lstsq(matrix, target, rcond=None)[0]
            residual = np.abs(design @ beta - local_depth)
            median = float(np.median(residual[active]))
            mad = float(np.median(np.abs(residual[active] - median)))
            cutoff = max(0.0025, median + 3.0 * 1.4826 * mad)
            next_active = residual <= cutoff
            if np.array_equal(next_active, active):
                break
            active = next_active
        inlier_count = int(active.sum())
        residual_values = residual[active]
        residual_p90 = (
            float(np.percentile(residual_values, 90.0))
            if len(residual_values)
            else float("inf")
        )
        candidate = {
            "passed": bool(
                inlier_count >= int(limits.min_depth_surface_inliers)
                and np.isfinite(beta[2])
                and residual_p90
                <= float(limits.max_depth_surface_residual_p90_m)
            ),
            "radius_px": float(radius),
            "candidate_samples": int(len(local_points)),
            "inlier_samples": inlier_count,
            "depth_m": float(beta[2]),
            "residual_p90_m": residual_p90,
            "nearest_support_px": float(np.min(local_distance)),
            "plane_coefficients": beta.astype(float).tolist(),
        }
        best = candidate
        if candidate["passed"]:
            break
    if best is None:
        best = {
            "passed": False,
            "radius_px": None,
            "candidate_samples": int(len(points)),
            "inlier_samples": 0,
            "depth_m": float("nan"),
            "residual_p90_m": float("inf"),
            "nearest_support_px": (
                float(np.min(distance)) if len(distance) else float("inf")
            ),
            "plane_coefficients": [float("nan")] * 3,
        }
    issues = []
    if best["inlier_samples"] < int(limits.min_depth_surface_inliers):
        issues.append("insufficient_local_depth_surface_inliers")
    if (
        not np.isfinite(best["residual_p90_m"])
        or best["residual_p90_m"]
        > float(limits.max_depth_surface_residual_p90_m)
    ):
        issues.append("local_depth_surface_residual_exceeded")
    if not np.isfinite(best["depth_m"]):
        issues.append("local_depth_surface_non_finite")
    best["issues"] = issues
    best["passed"] = not issues
    return best


def build_image_consistent_semantic_observations(
    semantic_points_by_view: Mapping[str, Mapping[str, Any]],
    dense_matches_by_view: Mapping[str, Mapping[str, Any]],
    rig: ProfileRig,
    *,
    work_size: tuple[int, int] = (640, 480),
    thresholds: SemanticRefinementThresholds | None = None,
) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    """Estimate semantic 3D points from local dense depth surfaces."""
    if rig.reference_view != "front":
        raise ValueError("semantic refinement requires the front reference camera")
    limits = thresholds or SemanticRefinementThresholds()
    front_points = semantic_points_by_view.get("front", {})
    front_camera = rig.cameras_by_view["front"]
    observations: dict[str, dict[str, np.ndarray]] = {"front": {}}
    depth_samples = {
        side_view: _dense_depth_samples(
            dense_matches_by_view[side_view],
            front_camera,
            rig.cameras_by_view[side_view],
            work_size=work_size,
            thresholds=limits,
        )
        for side_view in ("left", "right")
        if side_view in dense_matches_by_view
    }
    evidence: dict[str, dict[str, Any]] = {}
    for name, front_point in front_points.items():
        anchor_work = _scale_point(
            front_point, front_camera.image_size, work_size
        )
        side_evidence: dict[str, Any] = {}
        valid_depths: list[tuple[str, dict[str, Any]]] = []
        for side_view in ("left", "right"):
            side_points = semantic_points_by_view.get(side_view, {})
            matches = dense_matches_by_view.get(side_view)
            samples = depth_samples.get(side_view)
            if name not in side_points or matches is None or samples is None:
                side_evidence[side_view] = {
                    "passed": False,
                    "issues": ["missing_side_detector_or_dense_matches"],
                }
                continue
            affine_result = refine_semantic_pair(
                front_point,
                side_points[name],
                matches["points_front_work"],
                matches["points_side_work"],
                matches["confidence"],
                front_camera,
                rig.cameras_by_view[side_view],
                work_size=work_size,
                thresholds=limits,
            )
            depth_result = estimate_local_depth_surface(
                anchor_work,
                samples[0],
                samples[1],
                samples[2],
                thresholds=limits,
            )
            side_evidence[side_view] = {
                "passed": depth_result["passed"],
                "issues": depth_result["issues"],
                "depth_surface": depth_result,
                "local_affine_semantic_check": affine_result,
            }
            if depth_result["passed"]:
                valid_depths.append((side_view, depth_result))

        point_issues: list[str] = []
        if not valid_depths:
            point_issues.append("no_supported_local_depth_surface")
        cross_side_delta = None
        if len(valid_depths) == 2:
            cross_side_delta = abs(
                float(valid_depths[0][1]["depth_m"])
                - float(valid_depths[1][1]["depth_m"])
            )
            if cross_side_delta > float(limits.max_cross_side_depth_delta_m):
                point_issues.append("cross_side_depth_delta_exceeded")

        estimated_point = np.full(3, np.nan, dtype=np.float64)
        estimated_depth = float("nan")
        if valid_depths and not point_issues:
            fit_weights = np.asarray(
                [
                    result["inlier_samples"]
                    / max(result["residual_p90_m"], 1e-3) ** 2
                    for _view, result in valid_depths
                ],
                dtype=np.float64,
            )
            depth_values = np.asarray(
                [result["depth_m"] for _view, result in valid_depths],
                dtype=np.float64,
            )
            estimated_depth = float(
                np.average(depth_values, weights=fit_weights)
            )
            normalized = cv2.undistortPoints(
                np.asarray(front_point, dtype=np.float64).reshape(1, 1, 2),
                front_camera.K,
                front_camera.dist,
            ).reshape(2)
            estimated_point = np.array(
                (
                    normalized[0] * estimated_depth,
                    normalized[1] * estimated_depth,
                    estimated_depth,
                ),
                dtype=np.float64,
            )
            supported_views = ["front", *(view for view, _result in valid_depths)]
            for view in supported_views:
                observations.setdefault(view, {})[name] = project_reference_point(
                    estimated_point,
                    view,
                    rig,
                )

        evidence[name] = {
            "passed": not point_issues,
            "issues": point_issues,
            "side_evidence": side_evidence,
            "estimated_depth_m": estimated_depth,
            "estimated_point_reference_m": estimated_point.astype(float).tolist(),
            "cross_side_depth_delta_m": cross_side_delta,
            "three_view_depth_validated": len(valid_depths) == 2,
            "supported_side_views": [view for view, _result in valid_depths],
        }
    return observations, {
        "method": "robust_local_depth_surface_from_exact_loftr_triangulation",
        "reprojection_is_constructed_validation": True,
        "work_size": list(work_size),
        "points": evidence,
    }

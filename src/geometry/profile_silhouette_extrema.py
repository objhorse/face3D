"""Profile-silhouette evidence for high-curvature facial extrema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    relative_camera_transform,
    scale_intrinsics,
)
from src.geometry.profile_triangulation import (
    ProfileRig,
    TriangulationThresholds,
    project_reference_point,
    triangulate_profile_point,
)


SUPPORTED_EXTREMA = ("nose_tip", "chin")


@dataclass(frozen=True)
class SilhouetteExtremumThresholds:
    work_size: tuple[int, int] = (640, 480)
    max_epipolar_distance_work_px: float = 5.0
    nose_prior_radius_work_px: float = 48.0
    chin_prior_radius_work_px: float = 58.0
    mask_variant_offsets_px: tuple[int, ...] = (-2, 0, 2)
    max_variant_spread_work_px: float = 5.0
    max_variant_depth_spread_m: float = 0.006
    max_cross_side_depth_delta_m: float = 0.010
    min_valid_mask_variants: int = 2
    min_depth_m: float = 0.12
    max_depth_m: float = 1.50


def _largest_binary_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == largest, dtype=np.uint8)


def _letterbox_parameters(
    image_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> tuple[float, int, int]:
    image_width, image_height = (int(value) for value in image_size)
    canvas_height, canvas_width = (int(value) for value in canvas_shape)
    if min(image_width, image_height, canvas_width, canvas_height) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(
        canvas_width / float(image_width),
        canvas_height / float(image_height),
    )
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    return float(scale), int(x_offset), int(y_offset)


def canvas_points_to_original(
    points: Any,
    image_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scale, x_offset, y_offset = _letterbox_parameters(image_size, canvas_shape)
    result = values.copy()
    result[:, 0] = (result[:, 0] - x_offset) / scale
    result[:, 1] = (result[:, 1] - y_offset) / scale
    return result


def original_points_to_canvas(
    points: Any,
    image_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scale, x_offset, y_offset = _letterbox_parameters(image_size, canvas_shape)
    result = values * scale
    result[:, 0] += x_offset
    result[:, 1] += y_offset
    return result


def _original_to_work(
    points: Any,
    image_size: tuple[int, int],
    work_size: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return values * np.asarray(
        (
            work_size[0] / float(image_size[0]),
            work_size[1] / float(image_size[1]),
        ),
        dtype=np.float64,
    )


def _work_to_original(
    points: Any,
    image_size: tuple[int, int],
    work_size: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return values * np.asarray(
        (
            image_size[0] / float(work_size[0]),
            image_size[1] / float(work_size[1]),
        ),
        dtype=np.float64,
    )


def _undistort_work_points(
    points: Any,
    camera: Camera,
    work_size: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    intrinsics = scale_intrinsics(camera.K, camera.image_size, work_size)
    return cv2.undistortPoints(
        values.reshape(-1, 1, 2),
        intrinsics,
        camera.dist,
        P=intrinsics,
    ).reshape(-1, 2)


def _fundamental_matrix_work(
    front_camera: Camera,
    side_camera: Camera,
    work_size: tuple[int, int],
) -> np.ndarray:
    front_intrinsics = scale_intrinsics(
        front_camera.K, front_camera.image_size, work_size
    )
    side_intrinsics = scale_intrinsics(
        side_camera.K, side_camera.image_size, work_size
    )
    rotation, translation = relative_camera_transform(front_camera, side_camera)
    tx, ty, tz = np.asarray(translation, dtype=np.float64).reshape(3)
    skew = np.array(
        ((0.0, -tz, ty), (tz, 0.0, -tx), (-ty, tx, 0.0)),
        dtype=np.float64,
    )
    return np.linalg.inv(side_intrinsics).T @ skew @ rotation @ np.linalg.inv(
        front_intrinsics
    )


def _point_line_distances(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a, b, c = np.asarray(line, dtype=np.float64).reshape(3)
    denominator = float(np.hypot(a, b))
    if denominator <= 1e-12:
        return np.full(len(values), np.inf, dtype=np.float64)
    return np.abs(values[:, 0] * a + values[:, 1] * b + c) / denominator


def _contour_curvature(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(values)
    if count < 7:
        return np.zeros(count, dtype=np.float64)
    stride = max(2, min(12, count // 150))
    previous = values[np.arange(count) - stride]
    following = values[(np.arange(count) + stride) % count]
    first = values - previous
    second = following - values
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-9)
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-9)
    return np.clip(1.0 - np.sum(first * second, axis=1), 0.0, 2.0)


def _mask_variant(mask: np.ndarray, offset_px: int) -> np.ndarray:
    binary = _largest_binary_component(mask)
    amount = abs(int(offset_px))
    if amount == 0:
        return binary
    kernel_size = 2 * amount + 1
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (kernel_size, kernel_size),
    )
    operation = cv2.MORPH_DILATE if offset_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(binary, operation, kernel)


def _profile_contour_work(
    mask: np.ndarray,
    camera: Camera,
    work_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    binary = _largest_binary_component(mask)
    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError("face mask has no external contour")
    contour_canvas = max(contours, key=cv2.contourArea).reshape(-1, 2)
    contour_original = canvas_points_to_original(
        contour_canvas,
        camera.image_size,
        binary.shape,
    )
    width, height = camera.image_size
    valid = (
        (contour_original[:, 0] >= 0.0)
        & (contour_original[:, 0] < width)
        & (contour_original[:, 1] >= 0.0)
        & (contour_original[:, 1] < height)
    )
    contour_original = contour_original[valid]
    contour_canvas = contour_canvas[valid]
    if len(contour_original) < 16:
        raise ValueError("face contour has too little valid support")
    contour_work_distorted = _original_to_work(
        contour_original,
        camera.image_size,
        work_size,
    )
    contour_work = _undistort_work_points(
        contour_work_distorted,
        camera,
        work_size,
    )
    return contour_canvas.astype(np.float64), contour_work


def _select_contour_candidate(
    contour_work: np.ndarray,
    front_anchor_work: np.ndarray,
    side_prior_work: np.ndarray,
    front_camera: Camera,
    side_camera: Camera,
    *,
    semantic_name: str,
    thresholds: SilhouetteExtremumThresholds,
) -> dict[str, Any]:
    if semantic_name not in SUPPORTED_EXTREMA:
        raise ValueError(f"unsupported silhouette extremum: {semantic_name}")
    fundamental = _fundamental_matrix_work(
        front_camera,
        side_camera,
        thresholds.work_size,
    )
    line = fundamental @ np.append(front_anchor_work, 1.0)
    epipolar_distance = _point_line_distances(contour_work, line)
    prior_distance = np.linalg.norm(contour_work - side_prior_work, axis=1)
    prior_radius = (
        thresholds.nose_prior_radius_work_px
        if semantic_name == "nose_tip"
        else thresholds.chin_prior_radius_work_px
    )
    eligible = prior_distance <= float(prior_radius)
    if not np.any(eligible):
        return {
            "passed": False,
            "issues": ["no_contour_support_in_semantic_roi"],
            "epipolar_line_work": line.astype(float).tolist(),
        }
    curvature = _contour_curvature(contour_work)
    curvature_scale = max(
        float(np.percentile(curvature[eligible], 90.0)),
        1e-6,
    )
    curvature_normalized = np.clip(curvature / curvature_scale, 0.0, 1.0)
    epipolar_normalized = epipolar_distance / max(
        float(thresholds.max_epipolar_distance_work_px),
        1e-6,
    )
    prior_normalized = prior_distance / max(float(prior_radius), 1e-6)
    if semantic_name == "nose_tip":
        score = (
            0.70 * epipolar_normalized
            + 0.30 * prior_normalized
            - 0.08 * curvature_normalized
        )
    else:
        # The chin contour is broad and nearly tangent to its epipolar line.
        # Curvature peaks move between nearby jaw corners after tiny mask
        # perturbations, so continuity with the semantic chin seed is the
        # stronger tie-breaker here.
        score = 0.50 * epipolar_normalized + 0.50 * prior_normalized
    score[~eligible] = np.inf
    index = int(np.argmin(score))
    issues: list[str] = []
    if epipolar_distance[index] > float(
        thresholds.max_epipolar_distance_work_px
    ):
        issues.append("silhouette_epipolar_distance_exceeded")
    return {
        "passed": not issues,
        "issues": issues,
        "selected_index": index,
        "selected_work_px": contour_work[index].astype(float).tolist(),
        "epipolar_distance_work_px": float(epipolar_distance[index]),
        "semantic_prior_distance_work_px": float(prior_distance[index]),
        "curvature_score": float(curvature_normalized[index]),
        "selection_score": float(score[index]),
        "epipolar_line_work": line.astype(float).tolist(),
        "eligible_contour_points": int(np.count_nonzero(eligible)),
        "total_contour_points": int(len(contour_work)),
    }


def _triangulate_front_side(
    front_anchor_original: np.ndarray,
    side_point_work: np.ndarray,
    rig: ProfileRig,
    side_view: str,
    thresholds: SilhouetteExtremumThresholds,
) -> dict[str, Any]:
    side_camera = rig.cameras_by_view[side_view]
    side_point_original = _work_to_original(
        side_point_work,
        side_camera.image_size,
        thresholds.work_size,
    ).reshape(2)
    triangle = triangulate_profile_point(
        {
            "front": np.asarray(front_anchor_original, dtype=np.float64).reshape(2),
            side_view: side_point_original,
        },
        rig,
        thresholds=TriangulationThresholds(
            max_reprojection_px=45.0,
            min_ray_angle_deg=3.0,
            max_pair_delta_m=0.020,
            min_depth_m=thresholds.min_depth_m,
            max_depth_m=thresholds.max_depth_m,
        ),
    )
    triangle["side_observation_undistorted_px"] = (
        side_point_original.astype(float).tolist()
    )
    return triangle


def refine_profile_extremum(
    front_anchor_undistorted_px: Any,
    side_prior_distorted_px: Any,
    face_mask: np.ndarray,
    rig: ProfileRig,
    side_view: str,
    semantic_name: str,
    *,
    thresholds: SilhouetteExtremumThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or SilhouetteExtremumThresholds()
    if side_view not in {"left", "right"}:
        raise ValueError("side_view must be left or right")
    front_camera = rig.cameras_by_view["front"]
    side_camera = rig.cameras_by_view[side_view]
    front_anchor_original = np.asarray(
        front_anchor_undistorted_px,
        dtype=np.float64,
    ).reshape(2)
    front_anchor_work = _original_to_work(
        front_anchor_original,
        front_camera.image_size,
        limits.work_size,
    ).reshape(2)
    side_prior_work_distorted = _original_to_work(
        side_prior_distorted_px,
        side_camera.image_size,
        limits.work_size,
    )
    side_prior_work = _undistort_work_points(
        side_prior_work_distorted,
        side_camera,
        limits.work_size,
    ).reshape(2)

    variants: list[dict[str, Any]] = []
    for offset in limits.mask_variant_offsets_px:
        variant_mask = _mask_variant(face_mask, int(offset))
        try:
            contour_canvas, contour_work = _profile_contour_work(
                variant_mask,
                side_camera,
                limits.work_size,
            )
            selection = _select_contour_candidate(
                contour_work,
                front_anchor_work,
                side_prior_work,
                front_camera,
                side_camera,
                semantic_name=semantic_name,
                thresholds=limits,
            )
            selected_index = selection.get("selected_index")
            triangle = None
            selected_canvas = None
            if selected_index is not None:
                selected_canvas = contour_canvas[int(selected_index)]
                triangle = _triangulate_front_side(
                    front_anchor_original,
                    np.asarray(selection["selected_work_px"], dtype=np.float64),
                    rig,
                    side_view,
                    limits,
                )
            variant_passed = bool(
                selection["passed"]
                and triangle is not None
                and triangle["passed"]
            )
            variants.append(
                {
                    "offset_px": int(offset),
                    "passed": variant_passed,
                    "selection": selection,
                    "triangulation": triangle,
                    "selected_canvas_px": (
                        None
                        if selected_canvas is None
                        else selected_canvas.astype(float).tolist()
                    ),
                }
            )
        except (ValueError, cv2.error) as exc:
            variants.append(
                {
                    "offset_px": int(offset),
                    "passed": False,
                    "issues": [f"contour_extraction_failed:{exc}"],
                }
            )

    valid_variants = [item for item in variants if item["passed"]]
    issues: list[str] = []
    if len(valid_variants) < int(limits.min_valid_mask_variants):
        issues.append("insufficient_stable_mask_variants")
    selected_work = np.asarray(
        [
            item["selection"]["selected_work_px"]
            for item in valid_variants
        ],
        dtype=np.float64,
    ).reshape(-1, 2)
    points_reference = np.asarray(
        [
            item["triangulation"]["point_reference_m"]
            for item in valid_variants
        ],
        dtype=np.float64,
    ).reshape(-1, 3)
    variant_spread = (
        float(
            np.max(
                np.linalg.norm(
                    selected_work - np.median(selected_work, axis=0),
                    axis=1,
                )
            )
        )
        if len(selected_work)
        else float("inf")
    )
    depth_spread = (
        float(np.ptp(points_reference[:, 2]))
        if len(points_reference)
        else float("inf")
    )
    if variant_spread > float(limits.max_variant_spread_work_px):
        issues.append("mask_variant_pixel_spread_exceeded")
    if depth_spread > float(limits.max_variant_depth_spread_m):
        issues.append("mask_variant_depth_spread_exceeded")

    center_variant = next(
        (
            item
            for item in valid_variants
            if int(item["offset_px"]) == 0
        ),
        valid_variants[0] if valid_variants else None,
    )
    point_reference = (
        np.median(points_reference, axis=0)
        if len(points_reference)
        else np.full(3, np.nan, dtype=np.float64)
    )
    return {
        "passed": not issues and center_variant is not None,
        "issues": issues,
        "semantic_name": semantic_name,
        "side_view": side_view,
        "source": "side_face_mask_epipolar_extremum",
        "front_anchor_undistorted_px": front_anchor_original.astype(float).tolist(),
        "side_prior_distorted_px": np.asarray(
            side_prior_distorted_px,
            dtype=np.float64,
        ).reshape(2).astype(float).tolist(),
        "side_prior_undistorted_work_px": side_prior_work.astype(float).tolist(),
        "point_reference_m": point_reference.astype(float).tolist(),
        "variant_pixel_spread_work_px": variant_spread,
        "variant_depth_spread_m": depth_spread,
        "valid_variant_count": int(len(valid_variants)),
        "variants": variants,
        "selected_center": center_variant,
    }


def _front_ray_point(
    front_anchor_undistorted_px: np.ndarray,
    depth_m: float,
    front_camera: Camera,
) -> np.ndarray:
    ray = np.linalg.inv(front_camera.K) @ np.append(
        np.asarray(front_anchor_undistorted_px, dtype=np.float64).reshape(2),
        1.0,
    )
    if abs(float(ray[2])) <= 1e-12:
        return np.full(3, np.nan, dtype=np.float64)
    return ray * (float(depth_m) / float(ray[2]))


def build_profile_silhouette_extrema(
    profile_report: Mapping[str, Any],
    face_masks_by_view: Mapping[str, np.ndarray],
    rig: ProfileRig,
    *,
    thresholds: SilhouetteExtremumThresholds | None = None,
) -> dict[str, Any]:
    limits = thresholds or SilhouetteExtremumThresholds()
    points: dict[str, Any] = {}
    accepted: dict[str, list[float]] = {}
    warnings: list[str] = []
    for semantic_name in SUPPORTED_EXTREMA:
        front_anchor = np.asarray(
            profile_report["undistorted_mediapipe_px"]["front"][semantic_name],
            dtype=np.float64,
        ).reshape(2)
        side_reports: dict[str, Any] = {}
        for side_view in ("left", "right"):
            side_prior = profile_report["detectors"][side_view][semantic_name][
                "mediapipe_px"
            ]
            side_reports[side_view] = refine_profile_extremum(
                front_anchor,
                side_prior,
                face_masks_by_view[side_view],
                rig,
                side_view,
                semantic_name,
                thresholds=limits,
            )
        valid_sides = [
            report for report in side_reports.values() if report["passed"]
        ]
        issues: list[str] = []
        point_candidates = np.asarray(
            [report["point_reference_m"] for report in valid_sides],
            dtype=np.float64,
        ).reshape(-1, 3)
        cross_side_depth_delta = (
            float(np.ptp(point_candidates[:, 2]))
            if len(point_candidates) >= 2
            else None
        )
        if not valid_sides:
            issues.append("no_valid_side_silhouette_support")
        if (
            cross_side_depth_delta is not None
            and cross_side_depth_delta
            > float(limits.max_cross_side_depth_delta_m)
        ):
            issues.append("cross_side_silhouette_depth_disagreement")
        if len(valid_sides) == 1:
            warnings.append(f"{semantic_name}:single_side_only")
        if len(point_candidates):
            depth = float(np.median(point_candidates[:, 2]))
            fused = _front_ray_point(
                front_anchor,
                depth,
                rig.cameras_by_view["front"],
            )
        else:
            fused = np.full(3, np.nan, dtype=np.float64)
        passed = bool(not issues and np.isfinite(fused).all())
        if passed:
            accepted[semantic_name] = fused.astype(float).tolist()
        projections = (
            {
                view: project_reference_point(fused, view, rig)
                .astype(float)
                .tolist()
                for view in ("left", "front", "right")
            }
            if np.isfinite(fused).all()
            else {}
        )
        points[semantic_name] = {
            "passed": passed,
            "issues": issues,
            "warnings": (
                ["single_side_only"] if len(valid_sides) == 1 else []
            ),
            "side_reports": side_reports,
            "valid_side_count": len(valid_sides),
            "cross_side_depth_delta_m": cross_side_depth_delta,
            "point_reference_m": fused.astype(float).tolist(),
            "reprojected_undistorted_px": projections,
        }

    old_points = profile_report.get("accepted_points_reference_m", {})
    comparison: dict[str, Any] = {}
    for name, point in accepted.items():
        old = old_points.get(name)
        comparison[name] = {
            "silhouette_depth_m": float(point[2]),
            "local_surface_depth_m": (
                None if old is None else float(old[2])
            ),
            "silhouette_minus_surface_mm": (
                None
                if old is None
                else float((float(point[2]) - float(old[2])) * 1000.0)
            ),
        }
    if "nose_tip" in accepted and "upper_lip" in old_points:
        comparison["nose_tip_minus_upper_lip_mm"] = float(
            (float(old_points["upper_lip"][2]) - float(accepted["nose_tip"][2]))
            * 1000.0
        )
    if "chin" in accepted and "mouth_center" in old_points:
        comparison["mouth_center_minus_chin_mm"] = float(
            (float(accepted["chin"][2]) - float(old_points["mouth_center"][2]))
            * 1000.0
        )
    passed_names = sorted(accepted)
    return {
        "version": 1,
        "audit_only": True,
        "method": "front_semantic_ray_plus_side_silhouette_extrema",
        "points": points,
        "accepted_points_reference_m": accepted,
        "comparison_to_local_depth_surface": comparison,
        "quality_gate": {
            "passed": all(name in accepted for name in SUPPORTED_EXTREMA),
            "accepted_points": passed_names,
            "missing_points": [
                name for name in SUPPORTED_EXTREMA if name not in accepted
            ],
            "warnings": warnings,
        },
        "thresholds": {
            "work_size": list(limits.work_size),
            "max_epipolar_distance_work_px": float(
                limits.max_epipolar_distance_work_px
            ),
            "max_variant_spread_work_px": float(
                limits.max_variant_spread_work_px
            ),
            "max_variant_depth_spread_m": float(
                limits.max_variant_depth_spread_m
            ),
            "max_cross_side_depth_delta_m": float(
                limits.max_cross_side_depth_delta_m
            ),
            "min_valid_mask_variants": int(
                limits.min_valid_mask_variants
            ),
        },
    }

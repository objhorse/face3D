"""Profile-silhouette evidence for high-curvature facial extrema."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import cv2
import numpy as np

from src.cross_view_geometry import Camera
from src.geometry.observation_coordinates import (
    ObservationCoordinates,
    canvas_points_to_original,
    contour_curvature,
    external_contour_work,
    fundamental_matrix_work,
    mask_variant_work,
    normalize_mask_to_work,
    original_points_to_canvas,
    original_points_to_work,
    point_line_distances,
    work_points_to_original,
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
    mask_variant_offsets_work_px: tuple[int, ...] = (-2, 0, 2)
    max_variant_spread_work_px: float = 5.0
    max_variant_depth_spread_m: float = 0.006
    max_cross_side_depth_delta_m: float = 0.010
    min_valid_mask_variants: int = 2
    min_depth_m: float = 0.12
    max_depth_m: float = 1.50


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
    front_coordinates = ObservationCoordinates.from_camera(
        front_camera,
        work_size=thresholds.work_size,
        pixel_frame="undistorted",
    )
    side_coordinates = ObservationCoordinates.from_camera(
        side_camera,
        work_size=thresholds.work_size,
        pixel_frame="distorted",
    )
    fundamental = fundamental_matrix_work(
        front_camera,
        side_camera,
        front_coordinates,
        side_coordinates,
    )
    line = fundamental @ np.append(front_anchor_work, 1.0)
    epipolar_distance = point_line_distances(contour_work, line)
    prior_distance = np.linalg.norm(contour_work - side_prior_work, axis=1)
    prior_radius = (
        thresholds.nose_prior_radius_work_px
        if semantic_name == "nose_tip"
        else thresholds.chin_prior_radius_work_px
    )
    prior_eligible = prior_distance <= float(prior_radius)
    if not np.any(prior_eligible):
        return {
            "passed": False,
            "issues": ["no_contour_support_in_semantic_roi"],
            "epipolar_line_work": line.astype(float).tolist(),
        }
    eligible = prior_eligible & (
        epipolar_distance
        <= float(thresholds.max_epipolar_distance_work_px)
    )
    if not np.any(eligible):
        return {
            "passed": False,
            "issues": ["no_legal_contour_support_on_epipolar_line"],
            "epipolar_line_work": line.astype(float).tolist(),
            "prior_eligible_contour_points": int(
                np.count_nonzero(prior_eligible)
            ),
        }
    curvature = contour_curvature(contour_work)
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
    return {
        "passed": True,
        "issues": [],
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
    side_coordinates = ObservationCoordinates.from_camera(
        side_camera,
        work_size=thresholds.work_size,
        pixel_frame="distorted",
    )
    side_point_original = work_points_to_original(
        side_point_work,
        side_coordinates,
        target_pixel_frame="undistorted",
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
    front_coordinates = ObservationCoordinates.from_camera(
        front_camera,
        work_size=limits.work_size,
        pixel_frame="undistorted",
    )
    side_coordinates = ObservationCoordinates.from_camera(
        side_camera,
        work_size=limits.work_size,
        pixel_frame="distorted",
    )
    front_anchor_original = np.asarray(
        front_anchor_undistorted_px,
        dtype=np.float64,
    ).reshape(2)
    front_anchor_work = original_points_to_work(
        front_anchor_original,
        front_coordinates,
    ).reshape(2)
    side_prior_work = original_points_to_work(
        side_prior_distorted_px,
        side_coordinates,
    ).reshape(2)

    try:
        base_mask_work = normalize_mask_to_work(
            face_mask,
            side_coordinates,
            name="face mask",
        )
        mask_error = None
    except ValueError as exc:
        base_mask_work = None
        mask_error = str(exc)
    variants: list[dict[str, Any]] = []
    for offset in limits.mask_variant_offsets_work_px:
        if base_mask_work is None:
            variants.append(
                {
                    "offset_work_px": int(offset),
                    "passed": False,
                    "issues": [f"contour_extraction_failed:{mask_error}"],
                }
            )
            continue
        try:
            variant_mask = mask_variant_work(base_mask_work, int(offset))
            contour_work = external_contour_work(
                variant_mask,
                name="face mask",
                spacing_work_px=1.0,
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
                selected_original = work_points_to_original(
                    np.asarray(
                        selection["selected_work_px"],
                        dtype=np.float64,
                    ),
                    side_coordinates,
                    target_pixel_frame="distorted",
                )
                selected_canvas = original_points_to_canvas(
                    selected_original,
                    side_camera.image_size,
                    np.asarray(face_mask).shape[:2],
                ).reshape(2)
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
                    "offset_work_px": int(offset),
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
                    "offset_work_px": int(offset),
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
            if int(item["offset_work_px"]) == 0
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
            "mask_variant_offsets_work_px": list(
                limits.mask_variant_offsets_work_px
            ),
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

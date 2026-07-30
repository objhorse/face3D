"""Multi-dataset refinement of a fixed calibrated camera rig."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    relative_camera_transform,
    scale_intrinsics,
    triangulate_correspondences,
)
from src.learned_cross_view_geometry import (
    percentile_summary,
    rectified_vertical_error_for_pose,
    rotation_delta_degrees,
    translation_direction_delta_degrees,
)


@dataclass(frozen=True)
class MatchSet:
    """One dataset's pixel correspondences for a fixed camera pair."""

    dataset: str
    points_a: np.ndarray
    points_b: np.ndarray
    confidence: np.ndarray


@dataclass(frozen=True)
class JointRigThresholds:
    """Acceptance thresholds at the LoFTR work resolution."""

    min_datasets: int = 2
    min_matches_per_dataset: int = 100
    max_matches_per_dataset: int = 320
    min_essential_inliers: int = 80
    max_current_rotation_delta_deg: float = 15.0
    max_current_translation_delta_deg: float = 15.0
    max_validation_p50_px: float = 2.0
    max_validation_p90_px: float = 5.0
    min_validation_consensus_ratio: float = 0.75
    max_holdout_p50_px: float = 4.0
    max_holdout_p90_px: float = 6.0
    min_holdout_consensus_ratio: float = 0.65
    max_leave_one_out_rotation_delta_deg: float = 5.0
    max_leave_one_out_translation_delta_deg: float = 7.0
    min_common_triplets_per_dataset: int = 5
    min_common_triplets_total: int = 12
    max_common_front_distance_px: float = 0.75
    max_triplet_epipolar_px: float = 5.0
    max_depth_ratio_mad: float = 0.05
    max_dataset_depth_ratio_spread: float = 0.06
    max_baseline_scale_delta_ratio: float = 0.12


def _validated(match_set: MatchSet) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_a = np.asarray(match_set.points_a, dtype=np.float64)
    points_b = np.asarray(match_set.points_b, dtype=np.float64)
    confidence = np.asarray(match_set.confidence, dtype=np.float64).reshape(-1)
    if points_a.ndim != 2 or points_a.shape[1] != 2:
        raise ValueError(f"{match_set.dataset}: points_a must have shape (N, 2)")
    if points_b.shape != points_a.shape:
        raise ValueError(f"{match_set.dataset}: point arrays must have equal shape")
    if len(confidence) != len(points_a):
        raise ValueError(f"{match_set.dataset}: confidence length does not match points")
    finite = (
        np.isfinite(points_a).all(axis=1)
        & np.isfinite(points_b).all(axis=1)
        & np.isfinite(confidence)
    )
    return points_a[finite], points_b[finite], confidence[finite]


def _spatially_balanced_indices(
    points: np.ndarray,
    confidence: np.ndarray,
    *,
    limit: int,
    work_size: tuple[int, int],
    grid_size: tuple[int, int] = (8, 6),
) -> np.ndarray:
    if limit <= 0 or len(points) <= limit:
        return np.arange(len(points), dtype=np.int64)
    width, height = work_size
    grid_x, grid_y = grid_size
    order = np.argsort(-confidence, kind="stable")
    cell_limit = max(1, int(np.ceil(limit / float(grid_x * grid_y))))
    cell_counts: dict[tuple[int, int], int] = {}
    selected: list[int] = []
    selected_set: set[int] = set()
    for index in order:
        x, y = points[index]
        cell = (
            int(np.clip(np.floor(x / max(width, 1) * grid_x), 0, grid_x - 1)),
            int(np.clip(np.floor(y / max(height, 1) * grid_y), 0, grid_y - 1)),
        )
        if cell_counts.get(cell, 0) >= cell_limit:
            continue
        selected.append(int(index))
        selected_set.add(int(index))
        cell_counts[cell] = cell_counts.get(cell, 0) + 1
        if len(selected) == limit:
            break
    if len(selected) < limit:
        for index in order:
            value = int(index)
            if value in selected_set:
                continue
            selected.append(value)
            if len(selected) == limit:
                break
    return np.asarray(selected, dtype=np.int64)


def balanced_correspondences(
    match_sets: Sequence[MatchSet],
    *,
    max_matches_per_dataset: int,
    work_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack spatially distributed matches without letting one dataset dominate."""
    points_a_parts: list[np.ndarray] = []
    points_b_parts: list[np.ndarray] = []
    labels: list[str] = []
    for match_set in match_sets:
        points_a, points_b, confidence = _validated(match_set)
        indices = _spatially_balanced_indices(
            points_a,
            confidence,
            limit=int(max_matches_per_dataset),
            work_size=work_size,
        )
        points_a_parts.append(points_a[indices])
        points_b_parts.append(points_b[indices])
        labels.extend([str(match_set.dataset)] * len(indices))
    if not points_a_parts:
        raise ValueError("joint rig refinement requires at least one match set")
    return (
        np.concatenate(points_a_parts, axis=0),
        np.concatenate(points_b_parts, axis=0),
        np.asarray(labels, dtype=object),
    )


def _normalized_points(
    points: np.ndarray,
    camera: Camera,
    work_size: tuple[int, int],
) -> np.ndarray:
    intrinsics = scale_intrinsics(camera.K, camera.image_size, work_size)
    return cv2.undistortPoints(
        np.asarray(points, dtype=np.float64).reshape(-1, 1, 2),
        intrinsics,
        camera.dist,
    ).reshape(-1, 2)


def _recover_best_pose(
    essential: np.ndarray,
    points_a: np.ndarray,
    points_b: np.ndarray,
    inlier_mask: np.ndarray,
) -> tuple[int, np.ndarray, np.ndarray]:
    matrix = np.asarray(essential, dtype=np.float64)
    candidates = (
        [matrix]
        if matrix.shape == (3, 3)
        else [matrix[index : index + 3] for index in range(0, len(matrix), 3)]
    )
    best: tuple[int, np.ndarray, np.ndarray] | None = None
    for candidate in candidates:
        count, rotation, translation, _ = cv2.recoverPose(
            candidate,
            points_a,
            points_b,
            np.eye(3),
            mask=np.asarray(inlier_mask, dtype=np.uint8).copy(),
        )
        result = (
            int(count),
            np.asarray(rotation, dtype=np.float64),
            np.asarray(translation, dtype=np.float64).reshape(3),
        )
        if best is None or result[0] > best[0]:
            best = result
    if best is None:
        raise RuntimeError("essential pose recovery produced no candidates")
    return best


def _consensus_mask(absolute_error: np.ndarray) -> tuple[np.ndarray, float]:
    values = np.asarray(absolute_error, dtype=np.float64).reshape(-1)
    if not len(values):
        return np.zeros(0, dtype=bool), 0.0
    median = float(np.median(values))
    mad = float(np.median(np.abs(values - median)))
    robust_sigma = 1.4826 * mad
    threshold = float(np.clip(median + 3.0 * robust_sigma, 2.0, 5.0))
    return values <= threshold, threshold


def _estimate_from_normalized(
    normalized_a: np.ndarray,
    normalized_b: np.ndarray,
    baseline_translation: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int]:
    cv2.setRNGSeed(0)
    essential, inlier_mask = cv2.findEssentialMat(
        normalized_a,
        normalized_b,
        np.eye(3),
        method=cv2.RANSAC,
        prob=0.999,
        threshold=0.004,
    )
    if essential is None or inlier_mask is None or int(inlier_mask.sum()) < 8:
        raise RuntimeError("joint essential matrix estimation failed")
    inliers, rotation, direction = _recover_best_pose(
        essential,
        normalized_a,
        normalized_b,
        inlier_mask,
    )
    if float(np.dot(direction, baseline_translation)) < 0.0:
        direction = -direction
    baseline_scale = float(np.linalg.norm(baseline_translation))
    if baseline_scale <= 1e-9:
        raise ValueError("baseline camera translation has zero length")
    translation = direction / max(float(np.linalg.norm(direction)), 1e-12)
    translation *= baseline_scale
    return rotation, translation, int(inliers)


def _fit_relative_pose(
    match_sets: Sequence[MatchSet],
    camera_a: Camera,
    camera_b: Camera,
    *,
    max_matches_per_dataset: int,
    work_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, int, int]:
    points_a, points_b, _labels = balanced_correspondences(
        match_sets,
        max_matches_per_dataset=max_matches_per_dataset,
        work_size=work_size,
    )
    normalized_a = _normalized_points(points_a, camera_a, work_size)
    normalized_b = _normalized_points(points_b, camera_b, work_size)
    _baseline_rotation, baseline_translation = relative_camera_transform(
        camera_a,
        camera_b,
    )
    active = np.ones(len(points_a), dtype=bool)
    candidates: list[
        tuple[float, np.ndarray, np.ndarray, int, int]
    ] = []
    for _iteration in range(3):
        rotation, translation, inliers = _estimate_from_normalized(
            normalized_a[active],
            normalized_b[active],
            baseline_translation,
        )
        _signed, absolute_error = rectified_vertical_error_for_pose(
            points_a,
            points_b,
            camera_a,
            camera_b,
            rotation,
            translation,
        )
        consensus, _threshold = _consensus_mask(absolute_error)
        median = float(np.median(absolute_error))
        p90 = float(np.percentile(absolute_error, 90.0))
        consensus_ratio = float(consensus.mean())
        score = median + 0.25 * p90 + 2.0 * (1.0 - consensus_ratio)
        candidates.append(
            (
                score,
                rotation,
                translation,
                int(inliers),
                int(active.sum()),
            )
        )
        next_active = active & consensus
        if int(next_active.sum()) < 20 or np.array_equal(next_active, active):
            break
        active = next_active
    if not candidates:
        raise RuntimeError("joint pose refinement produced no pose")
    _score, rotation, translation, inliers, fitted_matches = min(
        candidates,
        key=lambda item: item[0],
    )
    return (
        rotation,
        translation,
        int(inliers),
        int(fitted_matches),
    )


def _evaluate_pose(
    match_set: MatchSet,
    camera_a: Camera,
    camera_b: Camera,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> dict[str, Any]:
    points_a, points_b, _confidence = _validated(match_set)
    signed_error, absolute_error = rectified_vertical_error_for_pose(
        points_a,
        points_b,
        camera_a,
        camera_b,
        rotation,
        translation,
    )
    consensus, consensus_threshold = _consensus_mask(absolute_error)
    consensus_error = absolute_error[consensus]
    return {
        "matches": int(len(points_a)),
        "absolute_y_error_px": percentile_summary(
            absolute_error,
            (50.0, 75.0, 90.0, 95.0),
        ),
        "signed_y_error_px": percentile_summary(
            signed_error,
            (5.0, 25.0, 50.0, 75.0, 95.0),
        ),
        "matches_within_3px": int((absolute_error <= 3.0).sum()),
        "consensus": {
            "threshold_px": consensus_threshold,
            "matches": int(consensus.sum()),
            "ratio": float(consensus.mean()) if len(consensus) else 0.0,
            "absolute_y_error_px": percentile_summary(
                consensus_error,
                (50.0, 75.0, 90.0, 95.0),
            ),
        },
    }


def _max_pose_spread(
    poses: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[float, float]:
    if len(poses) < 2:
        return 0.0, 0.0
    rotation_spread = max(
        rotation_delta_degrees(first[0], second[0])
        for first, second in combinations(poses, 2)
    )
    translation_spread = max(
        translation_direction_delta_degrees(first[1], second[1])
        for first, second in combinations(poses, 2)
    )
    return float(rotation_spread), float(translation_spread)


def _mutual_nearest_pairs(
    first: np.ndarray,
    second: np.ndarray,
    *,
    max_distance: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(first) or not len(second):
        empty = np.empty(0, dtype=np.int64)
        return empty, empty, np.empty(0, dtype=np.float64)
    distances = np.linalg.norm(
        np.asarray(first, dtype=np.float64)[:, None, :]
        - np.asarray(second, dtype=np.float64)[None, :, :],
        axis=2,
    )
    second_for_first = np.argmin(distances, axis=1)
    first_for_second = np.argmin(distances, axis=0)
    first_indices = np.asarray(
        [
            index
            for index, candidate in enumerate(second_for_first)
            if first_for_second[candidate] == index
            and distances[index, candidate] <= float(max_distance)
        ],
        dtype=np.int64,
    )
    second_indices = second_for_first[first_indices].astype(np.int64)
    return (
        first_indices,
        second_indices,
        distances[first_indices, second_indices],
    )


def _triangulated_depths(
    match_set: MatchSet,
    camera_a: Camera,
    camera_b: Camera,
    rotation: np.ndarray,
    translation: np.ndarray,
    *,
    work_size: tuple[int, int],
    max_epipolar_error_px: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points_a, points_b, _confidence = _validated(match_set)
    intrinsics_a = scale_intrinsics(camera_a.K, camera_a.image_size, work_size)
    intrinsics_b = scale_intrinsics(camera_b.K, camera_b.image_size, work_size)
    undistorted_a = cv2.undistortPoints(
        points_a.reshape(-1, 1, 2),
        intrinsics_a,
        camera_a.dist,
        P=intrinsics_a,
    ).reshape(-1, 2)
    undistorted_b = cv2.undistortPoints(
        points_b.reshape(-1, 1, 2),
        intrinsics_b,
        camera_b.dist,
        P=intrinsics_b,
    ).reshape(-1, 2)
    xyz, positive = triangulate_correspondences(
        undistorted_a,
        undistorted_b,
        intrinsics_a,
        intrinsics_b,
        rotation,
        translation,
    )
    _signed, epipolar_error = rectified_vertical_error_for_pose(
        points_a,
        points_b,
        camera_a,
        camera_b,
        rotation,
        translation,
    )
    valid = (
        positive
        & np.isfinite(xyz[:, 2])
        & (xyz[:, 2] > 0.0)
        & (epipolar_error <= float(max_epipolar_error_px))
    )
    return points_a, xyz[:, 2], valid


def reconcile_triplet_baseline_scales(
    front_left_matches: Sequence[MatchSet],
    front_right_matches: Sequence[MatchSet],
    front_camera: Camera,
    left_camera: Camera,
    right_camera: Camera,
    left_rotation: np.ndarray,
    left_translation: np.ndarray,
    right_rotation: np.ndarray,
    right_translation: np.ndarray,
    *,
    thresholds: JointRigThresholds | None = None,
    work_size: tuple[int, int] = (640, 480),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Recover the left/right baseline ratio from common front-image matches."""
    limits = thresholds or JointRigThresholds()
    left_by_dataset = {item.dataset: item for item in front_left_matches}
    right_by_dataset = {item.dataset: item for item in front_right_matches}
    datasets = sorted(set(left_by_dataset) & set(right_by_dataset))
    issues: list[str] = []
    ratios_by_dataset: dict[str, np.ndarray] = {}
    dataset_metrics: dict[str, Any] = {}
    for dataset in datasets:
        left_front, left_depth, left_valid = _triangulated_depths(
            left_by_dataset[dataset],
            front_camera,
            left_camera,
            left_rotation,
            left_translation,
            work_size=work_size,
            max_epipolar_error_px=limits.max_triplet_epipolar_px,
        )
        right_front, right_depth, right_valid = _triangulated_depths(
            right_by_dataset[dataset],
            front_camera,
            right_camera,
            right_rotation,
            right_translation,
            work_size=work_size,
            max_epipolar_error_px=limits.max_triplet_epipolar_px,
        )
        left_indices, right_indices, front_distance = _mutual_nearest_pairs(
            left_front,
            right_front,
            max_distance=limits.max_common_front_distance_px,
        )
        valid = (
            left_valid[left_indices]
            & right_valid[right_indices]
            & np.isfinite(left_depth[left_indices])
            & np.isfinite(right_depth[right_indices])
            & (right_depth[right_indices] > 1e-9)
        )
        ratios = left_depth[left_indices[valid]] / right_depth[right_indices[valid]]
        ratios = ratios[np.isfinite(ratios) & (ratios > 0.5) & (ratios < 2.0)]
        ratios_by_dataset[dataset] = ratios
        median = float(np.median(ratios)) if len(ratios) else float("nan")
        mad = (
            float(np.median(np.abs(ratios - median)))
            if len(ratios)
            else float("inf")
        )
        dataset_metrics[dataset] = {
            "common_front_matches": int(len(left_indices)),
            "valid_triplets": int(len(ratios)),
            "front_match_distance_px": percentile_summary(
                front_distance[valid],
                (50.0, 90.0),
            ),
            "left_over_right_depth_ratio": percentile_summary(
                ratios,
                (10.0, 25.0, 50.0, 75.0, 90.0),
            ),
            "depth_ratio_mad": mad,
        }
        if len(ratios) < int(limits.min_common_triplets_per_dataset):
            issues.append(f"insufficient_common_triplets:{dataset}")
        if mad > float(limits.max_depth_ratio_mad):
            issues.append(f"triplet_depth_ratio_unstable:{dataset}")

    populated = [values for values in ratios_by_dataset.values() if len(values)]
    pooled = (
        np.concatenate(populated)
        if populated
        else np.empty(0, dtype=np.float64)
    )
    if len(pooled) < int(limits.min_common_triplets_total):
        issues.append("insufficient_common_triplets_total")
    pooled_median = float(np.median(pooled)) if len(pooled) else float("nan")
    pooled_mad = (
        float(np.median(np.abs(pooled - pooled_median)))
        if len(pooled)
        else float("inf")
    )
    dataset_medians = [
        float(metrics["left_over_right_depth_ratio"]["p50"])
        for metrics in dataset_metrics.values()
        if metrics["left_over_right_depth_ratio"]["p50"] is not None
    ]
    dataset_spread = (
        max(dataset_medians) - min(dataset_medians)
        if len(dataset_medians) >= 2
        else 0.0
    )
    if pooled_mad > float(limits.max_depth_ratio_mad):
        issues.append("pooled_triplet_depth_ratio_unstable")
    if dataset_spread > float(limits.max_dataset_depth_ratio_spread):
        issues.append("triplet_depth_ratio_cross_dataset_spread_exceeded")

    left_length = float(np.linalg.norm(left_translation))
    right_length = float(np.linalg.norm(right_translation))
    current_ratio = right_length / max(left_length, 1e-12)
    target_ratio = current_ratio * pooled_median
    denominator = (
        1.0 / max(left_length * left_length, 1e-12)
        + target_ratio * target_ratio
        / max(right_length * right_length, 1e-12)
    )
    adjusted_left_length = (
        1.0 / max(left_length, 1e-12)
        + target_ratio / max(right_length, 1e-12)
    ) / denominator
    adjusted_right_length = target_ratio * adjusted_left_length
    left_scale_delta = adjusted_left_length / max(left_length, 1e-12) - 1.0
    right_scale_delta = adjusted_right_length / max(right_length, 1e-12) - 1.0
    values = np.asarray(
        (target_ratio, adjusted_left_length, adjusted_right_length),
        dtype=np.float64,
    )
    if not np.isfinite(values).all():
        issues.append("baseline_scale_reconciliation_non_finite")
    if max(abs(left_scale_delta), abs(right_scale_delta)) > float(
        limits.max_baseline_scale_delta_ratio
    ):
        issues.append("baseline_scale_delta_exceeded")

    adjusted_left = (
        np.asarray(left_translation, dtype=np.float64)
        / max(left_length, 1e-12)
        * adjusted_left_length
    )
    adjusted_right = (
        np.asarray(right_translation, dtype=np.float64)
        / max(right_length, 1e-12)
        * adjusted_right_length
    )
    unique_issues = list(dict.fromkeys(issues))
    return (
        {
            "accepted": not unique_issues,
            "issues": unique_issues,
            "datasets": dataset_metrics,
            "valid_triplets_total": int(len(pooled)),
            "pooled_left_over_right_depth_ratio": percentile_summary(
                pooled,
                (10.0, 25.0, 50.0, 75.0, 90.0),
            ),
            "pooled_depth_ratio_mad": pooled_mad,
            "dataset_median_spread": float(dataset_spread),
            "current_right_over_left_baseline_ratio": current_ratio,
            "target_right_over_left_baseline_ratio": target_ratio,
            "current_baseline_m": {
                "left": left_length,
                "right": right_length,
            },
            "adjusted_baseline_m": {
                "left": adjusted_left_length,
                "right": adjusted_right_length,
            },
            "baseline_scale_delta_ratio": {
                "left": left_scale_delta,
                "right": right_scale_delta,
            },
        },
        adjusted_left,
        adjusted_right,
    )


def refine_joint_pair(
    match_sets: Sequence[MatchSet],
    camera_a: Camera,
    camera_b: Camera,
    *,
    thresholds: JointRigThresholds | None = None,
    work_size: tuple[int, int] = (640, 480),
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    """Estimate one fixed pair pose and audit it on every held-out dataset."""
    limits = thresholds or JointRigThresholds()
    if len(match_sets) < int(limits.min_datasets):
        raise ValueError(
            f"joint rig refinement requires {limits.min_datasets} datasets"
        )
    names = [str(item.dataset) for item in match_sets]
    if len(set(names)) != len(names):
        raise ValueError("joint rig dataset names must be unique")

    issues: list[str] = []
    warnings: list[str] = []
    counts: dict[str, int] = {}
    for match_set in match_sets:
        points_a, _points_b, _confidence = _validated(match_set)
        counts[match_set.dataset] = int(len(points_a))
        if len(points_a) < int(limits.min_matches_per_dataset):
            issues.append(f"insufficient_matches:{match_set.dataset}")

    rotation, translation, inliers, fitted_matches = _fit_relative_pose(
        match_sets,
        camera_a,
        camera_b,
        max_matches_per_dataset=int(limits.max_matches_per_dataset),
        work_size=work_size,
    )
    baseline_rotation, baseline_translation = relative_camera_transform(
        camera_a,
        camera_b,
    )
    rotation_delta = rotation_delta_degrees(rotation, baseline_rotation)
    translation_delta = translation_direction_delta_degrees(
        translation,
        baseline_translation,
    )
    if inliers < int(limits.min_essential_inliers):
        issues.append("insufficient_joint_essential_inliers")
    if rotation_delta > float(limits.max_current_rotation_delta_deg):
        issues.append("rotation_delta_exceeded")
    if translation_delta > float(limits.max_current_translation_delta_deg):
        issues.append("translation_delta_exceeded")

    validation: dict[str, Any] = {}
    for match_set in match_sets:
        metrics = _evaluate_pose(
            match_set,
            camera_a,
            camera_b,
            rotation,
            translation,
        )
        validation[match_set.dataset] = metrics
        summary = metrics["absolute_y_error_px"]
        consensus = metrics["consensus"]
        consensus_error = consensus["absolute_y_error_px"]
        if (
            summary["p50"] is None
            or float(summary["p50"]) > float(limits.max_validation_p50_px)
            or consensus_error["p90"] is None
            or float(consensus_error["p90"])
            > float(limits.max_validation_p90_px)
            or float(consensus["ratio"])
            < float(limits.min_validation_consensus_ratio)
        ):
            issues.append(f"validation_error_exceeded:{match_set.dataset}")

    leave_one_out: dict[str, Any] = {}
    leave_one_out_poses: list[tuple[np.ndarray, np.ndarray]] = []
    holdout_failed = False
    for held_out in match_sets:
        training = [item for item in match_sets if item.dataset != held_out.dataset]
        hold_rotation, hold_translation, hold_inliers, hold_matches = (
            _fit_relative_pose(
                training,
                camera_a,
                camera_b,
                max_matches_per_dataset=int(limits.max_matches_per_dataset),
                work_size=work_size,
            )
        )
        leave_one_out_poses.append((hold_rotation, hold_translation))
        metrics = _evaluate_pose(
            held_out,
            camera_a,
            camera_b,
            hold_rotation,
            hold_translation,
        )
        metrics.update(
            {
                "trained_on": [item.dataset for item in training],
                "essential_inliers": int(hold_inliers),
                "fitted_matches": int(hold_matches),
            }
        )
        leave_one_out[held_out.dataset] = metrics
        summary = metrics["absolute_y_error_px"]
        consensus = metrics["consensus"]
        consensus_error = consensus["absolute_y_error_px"]
        if (
            summary["p50"] is None
            or float(summary["p50"]) > float(limits.max_holdout_p50_px)
            or consensus_error["p90"] is None
            or float(consensus_error["p90"]) > float(limits.max_holdout_p90_px)
            or float(consensus["ratio"])
            < float(limits.min_holdout_consensus_ratio)
        ):
            issues.append(f"holdout_error_exceeded:{held_out.dataset}")
            holdout_failed = True

    rotation_spread, translation_spread = _max_pose_spread(
        leave_one_out_poses
    )
    if rotation_spread > float(limits.max_leave_one_out_rotation_delta_deg):
        message = "leave_one_out_pose_inconsistent:rotation"
        if holdout_failed:
            issues.append(message)
        else:
            warnings.append(f"{message}:cross_validation_passed")
    if translation_spread > float(
        limits.max_leave_one_out_translation_delta_deg
    ):
        message = "leave_one_out_pose_inconsistent:translation"
        if holdout_failed:
            issues.append(message)
        else:
            warnings.append(f"{message}:cross_validation_passed")

    all_errors = [
        validation[name]["absolute_y_error_px"]
        for name in names
    ]
    median_values = [float(item["p50"]) for item in all_errors if item["p50"] is not None]
    p90_values = [float(item["p90"]) for item in all_errors if item["p90"] is not None]
    matches_within = sum(
        int(validation[name]["matches_within_3px"]) for name in names
    )
    unique_issues = list(dict.fromkeys(issues))
    metrics: dict[str, Any] = {
        "accepted": not unique_issues,
        "issues": unique_issues,
        "warnings": list(dict.fromkeys(warnings)),
        "datasets": names,
        "matches_by_dataset": counts,
        "joint_essential_inliers": int(inliers),
        "joint_fitted_matches": int(fitted_matches),
        "rotation_delta_deg": float(rotation_delta),
        "translation_direction_delta_deg": float(translation_delta),
        "translation_scale_m": float(np.linalg.norm(translation)),
        "validation": validation,
        "leave_one_out": leave_one_out,
        "leave_one_out_pose_spread": {
            "rotation_deg": rotation_spread,
            "translation_direction_deg": translation_spread,
        },
        "thresholds": {
            key: value
            for key, value in vars(limits).items()
        },
        "refined_pose_candidate": {
            "accepted": not unique_issues,
            "essential_inliers": int(inliers),
            "rotation_delta_deg": float(rotation_delta),
            "translation_direction_delta_deg": float(translation_delta),
            "rig_y_error_px": {
                "p50": max(median_values, default=float("inf")),
                "p90": max(p90_values, default=float("inf")),
            },
            "matches_within_3px": int(matches_within),
        },
    }
    return metrics, rotation, translation

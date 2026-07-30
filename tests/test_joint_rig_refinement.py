from __future__ import annotations

import math

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.joint_rig_refinement import (
    JointRigThresholds,
    MatchSet,
    balanced_correspondences,
    reconcile_triplet_baseline_scales,
    refine_joint_pair,
)
from src.learned_cross_view_geometry import (
    rotation_delta_degrees,
    translation_direction_delta_degrees,
)


WORK_SIZE = (640, 480)


def _yaw(degrees: float) -> np.ndarray:
    angle = math.radians(degrees)
    return np.array(
        [
            [math.cos(angle), 0.0, math.sin(angle)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle), 0.0, math.cos(angle)],
        ],
        dtype=np.float64,
    )


def _camera(
    name: str,
    view: str,
    rotation: np.ndarray,
    translation: np.ndarray,
) -> Camera:
    return Camera(
        name=name,
        view=view,
        image_size=WORK_SIZE,
        K=np.array(
            [[520.0, 0.0, 320.0], [0.0, 515.0, 240.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=np.asarray(rotation, dtype=np.float64),
        t_rig_to_camera=np.asarray(translation, dtype=np.float64),
    )


def _project(points: np.ndarray, camera: Camera) -> np.ndarray:
    camera_points = (
        camera.R_rig_to_camera @ np.asarray(points, dtype=np.float64).T
    ).T + camera.t_rig_to_camera
    normalized = camera_points[:, :2] / camera_points[:, 2:3]
    return normalized @ camera.K[:2, :2].T + camera.K[:2, 2]


def _synthetic_matches(
    dataset: str,
    seed: int,
    camera_a: Camera,
    camera_b: Camera,
) -> MatchSet:
    rng = np.random.default_rng(seed)
    points = np.column_stack(
        [
            rng.uniform(-0.14, 0.14, 240),
            rng.uniform(-0.17, 0.17, 240),
            rng.uniform(0.55, 0.90, 240),
        ]
    )
    noise_a = rng.normal(0.0, 0.12, (len(points), 2))
    noise_b = rng.normal(0.0, 0.12, (len(points), 2))
    confidence = rng.uniform(0.5, 1.0, len(points))
    return MatchSet(
        dataset=dataset,
        points_a=_project(points, camera_a) + noise_a,
        points_b=_project(points, camera_b) + noise_b,
        confidence=confidence,
    )


def test_balanced_correspondences_caps_each_dataset() -> None:
    first = MatchSet(
        dataset="first",
        points_a=np.column_stack([np.arange(30), np.arange(30)]),
        points_b=np.column_stack([np.arange(30), np.arange(30)]),
        confidence=np.linspace(0.1, 1.0, 30),
    )
    second = MatchSet(
        dataset="second",
        points_a=np.column_stack([np.arange(12), np.arange(12)]),
        points_b=np.column_stack([np.arange(12), np.arange(12)]),
        confidence=np.linspace(0.2, 1.0, 12),
    )

    points_a, points_b, labels = balanced_correspondences(
        [first, second],
        max_matches_per_dataset=10,
        work_size=WORK_SIZE,
    )

    assert points_a.shape == (20, 2)
    assert points_b.shape == (20, 2)
    assert list(labels).count("first") == 10
    assert list(labels).count("second") == 10


def test_joint_refinement_recovers_fixed_pose_and_preserves_scale() -> None:
    reference = _camera("camera2", "front", np.eye(3), np.zeros(3))
    truth = _camera(
        "camera1",
        "left",
        _yaw(9.0),
        np.array([0.125, 0.004, 0.028]),
    )
    baseline = _camera(
        "camera1",
        "left",
        _yaw(2.0),
        np.array([0.118, -0.004, 0.043]),
    )
    matches = [
        _synthetic_matches("capture_a", 3, reference, truth),
        _synthetic_matches("capture_b", 11, reference, truth),
    ]
    thresholds = JointRigThresholds(
        max_matches_per_dataset=200,
        max_current_rotation_delta_deg=15.0,
        max_current_translation_delta_deg=20.0,
        max_validation_p50_px=1.0,
        max_validation_p90_px=2.0,
        max_holdout_p50_px=1.5,
        max_holdout_p90_px=3.0,
        max_leave_one_out_rotation_delta_deg=2.0,
        max_leave_one_out_translation_delta_deg=6.0,
    )

    metrics, rotation, translation = refine_joint_pair(
        matches,
        reference,
        baseline,
        thresholds=thresholds,
        work_size=WORK_SIZE,
    )

    assert metrics["accepted"] is True
    assert rotation_delta_degrees(rotation, truth.R_rig_to_camera) < 0.5
    assert (
        translation_direction_delta_degrees(
            translation,
            truth.t_rig_to_camera,
        )
        < 1.0
    )
    assert np.linalg.norm(translation) == pytest.approx(
        np.linalg.norm(baseline.t_rig_to_camera),
        rel=1e-8,
    )
    assert set(metrics["validation"]) == {"capture_a", "capture_b"}
    assert set(metrics["leave_one_out"]) == {"capture_a", "capture_b"}


def test_joint_refinement_rejects_dataset_specific_pose() -> None:
    reference = _camera("camera2", "front", np.eye(3), np.zeros(3))
    baseline = _camera(
        "camera1",
        "left",
        _yaw(4.0),
        np.array([0.12, 0.0, 0.03]),
    )
    first_truth = _camera(
        "camera1",
        "left",
        _yaw(7.0),
        np.array([0.12, 0.0, 0.03]),
    )
    second_truth = _camera(
        "camera1",
        "left",
        _yaw(15.0),
        np.array([0.08, 0.02, 0.09]),
    )
    matches = [
        _synthetic_matches("capture_a", 17, reference, first_truth),
        _synthetic_matches("capture_b", 23, reference, second_truth),
    ]

    metrics, _rotation, _translation = refine_joint_pair(
        matches,
        reference,
        baseline,
        thresholds=JointRigThresholds(
            max_matches_per_dataset=200,
            max_leave_one_out_rotation_delta_deg=2.0,
            max_leave_one_out_translation_delta_deg=4.0,
        ),
        work_size=WORK_SIZE,
    )

    assert metrics["accepted"] is False
    assert any(
        issue.startswith("leave_one_out_pose_inconsistent")
        for issue in metrics["issues"]
    )


def test_joint_refinement_uses_robust_consensus_for_match_outliers() -> None:
    reference = _camera("camera2", "front", np.eye(3), np.zeros(3))
    truth = _camera(
        "camera1",
        "left",
        _yaw(8.0),
        np.array([0.12, 0.003, 0.03]),
    )
    baseline = _camera(
        "camera1",
        "left",
        _yaw(2.0),
        np.array([0.115, -0.003, 0.045]),
    )
    rng = np.random.default_rng(31)
    matches = [
        _synthetic_matches("capture_a", 29, reference, truth),
        _synthetic_matches("capture_b", 37, reference, truth),
    ]
    contaminated: list[MatchSet] = []
    for match_set in matches:
        points_b = match_set.points_b.copy()
        bad = rng.choice(len(points_b), size=36, replace=False)
        points_b[bad] += rng.normal(0.0, 24.0, (len(bad), 2))
        contaminated.append(
            MatchSet(
                dataset=match_set.dataset,
                points_a=match_set.points_a,
                points_b=points_b,
                confidence=match_set.confidence,
            )
        )

    metrics, rotation, _translation = refine_joint_pair(
        contaminated,
        reference,
        baseline,
        thresholds=JointRigThresholds(
            max_matches_per_dataset=220,
            max_validation_p50_px=2.0,
            max_validation_p90_px=5.0,
            min_validation_consensus_ratio=0.75,
            max_holdout_p50_px=4.0,
            max_holdout_p90_px=6.0,
            min_holdout_consensus_ratio=0.65,
        ),
        work_size=WORK_SIZE,
    )

    assert metrics["accepted"] is True
    assert rotation_delta_degrees(rotation, truth.R_rig_to_camera) < 1.5
    assert all(
        result["consensus"]["ratio"] < 0.95
        for result in metrics["validation"].values()
    )


def test_triplet_reconciliation_recovers_relative_baseline_scale() -> None:
    front = _camera("camera2", "front", np.eye(3), np.zeros(3))
    left_truth = _camera(
        "camera1", "left", np.eye(3), np.array([-0.10, 0.0, 0.0])
    )
    right_truth = _camera(
        "camera3", "right", np.eye(3), np.array([0.12, 0.0, 0.0])
    )
    left_source = _camera(
        "camera1", "left", np.eye(3), np.array([-0.10, 0.0, 0.0])
    )
    right_source = _camera(
        "camera3", "right", np.eye(3), np.array([0.10, 0.0, 0.0])
    )
    left_matches: list[MatchSet] = []
    right_matches: list[MatchSet] = []
    for dataset, seed in (("capture_a", 43), ("capture_b", 47)):
        rng = np.random.default_rng(seed)
        points = np.column_stack(
            (
                rng.uniform(-0.12, 0.12, 80),
                rng.uniform(-0.15, 0.15, 80),
                rng.uniform(0.55, 0.85, 80),
            )
        )
        front_pixels = _project(points, front)
        confidence = np.full(len(points), 0.9, dtype=np.float64)
        left_matches.append(
            MatchSet(
                dataset,
                front_pixels,
                _project(points, left_truth),
                confidence,
            )
        )
        right_matches.append(
            MatchSet(
                dataset,
                front_pixels,
                _project(points, right_truth),
                confidence,
            )
        )

    metrics, adjusted_left, adjusted_right = reconcile_triplet_baseline_scales(
        left_matches,
        right_matches,
        front,
        left_source,
        right_source,
        np.eye(3),
        left_source.t_rig_to_camera,
        np.eye(3),
        right_source.t_rig_to_camera,
        thresholds=JointRigThresholds(
            min_common_triplets_per_dataset=20,
            min_common_triplets_total=60,
        ),
        work_size=WORK_SIZE,
    )

    assert metrics["accepted"] is True
    assert metrics["pooled_left_over_right_depth_ratio"]["p50"] == pytest.approx(
        1.2, rel=1e-3
    )
    assert np.linalg.norm(adjusted_right) / np.linalg.norm(
        adjusted_left
    ) == pytest.approx(1.2, rel=1e-3)
    assert max(
        abs(metrics["baseline_scale_delta_ratio"]["left"]),
        abs(metrics["baseline_scale_delta_ratio"]["right"]),
    ) < 0.12

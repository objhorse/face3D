from __future__ import annotations

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.profile_observations import (
    ObservationThresholds,
    evaluate_semantic_profile_observations,
    semantic_points_from_face_alignment,
    semantic_points_from_mediapipe,
    undistort_semantic_points,
)
from src.geometry.profile_triangulation import ProfileRig, project_reference_point


def _camera(name: str, view: str, center_x: float) -> Camera:
    rotation = np.eye(3, dtype=np.float64)
    center = np.array([center_x, 0.0, 0.0], dtype=np.float64)
    return Camera(
        name=name,
        view=view,
        image_size=(1024, 768),
        K=np.array(
            [[1000.0, 0.0, 512.0], [0.0, 1000.0, 384.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=rotation,
        t_rig_to_camera=-rotation @ center,
    )


def _rig() -> ProfileRig:
    return ProfileRig(
        cameras_by_view={
            "left": _camera("camera1", "left", -0.10),
            "front": _camera("camera2", "front", 0.0),
            "right": _camera("camera3", "right", 0.10),
        },
        reference_view="front",
        units="meters",
        calibration_path="synthetic",
        stereo_rms_px={"left": 0.1, "front": 0.0, "right": 0.1},
    )


def test_semantic_detector_definitions_use_anatomical_midline_indices():
    mediapipe = np.zeros((478, 2), dtype=np.float64)
    face_alignment = np.zeros((68, 2), dtype=np.float64)
    mediapipe[4] = (4.0, 40.0)
    mediapipe[13] = (13.0, 130.0)
    mediapipe[14] = (14.0, 140.0)
    face_alignment[30] = (30.0, 300.0)
    face_alignment[62] = (62.0, 620.0)
    face_alignment[66] = (66.0, 660.0)

    mp_points = semantic_points_from_mediapipe(mediapipe)
    fa_points = semantic_points_from_face_alignment(face_alignment)

    assert mp_points["nose_tip"] == pytest.approx((4.0, 40.0))
    assert mp_points["mouth_center"] == pytest.approx((13.5, 135.0))
    assert fa_points["nose_tip"] == pytest.approx((30.0, 300.0))
    assert fa_points["mouth_center"] == pytest.approx((64.0, 640.0))


def test_zero_distortion_preserves_original_resolution_pixels():
    camera = _camera("camera2", "front", 0.0)
    points = {"nose_tip": np.array([620.0, 410.0])}

    result = undistort_semantic_points(points, camera)

    assert result["nose_tip"] == pytest.approx(points["nose_tip"], abs=1e-9)


def _semantic_observations(rig: ProfileRig):
    points = {
        "nose_tip": np.array([0.0, -0.025, 0.48]),
        "subnasale": np.array([0.0, 0.000, 0.50]),
        "upper_lip": np.array([0.0, 0.022, 0.51]),
        "lower_lip": np.array([0.0, 0.038, 0.512]),
        "mouth_center": np.array([0.0, 0.030, 0.511]),
        "chin": np.array([0.0, 0.095, 0.525]),
    }
    observations = {
        view: {
            name: project_reference_point(point, view, rig)
            for name, point in points.items()
        }
        for view in ("left", "front", "right")
    }
    return points, observations


def test_consistent_three_view_semantics_pass_profile_gate():
    rig = _rig()
    expected, observations = _semantic_observations(rig)

    report = evaluate_semantic_profile_observations(
        observations,
        observations,
        rig,
        {view: (768, 1024, 3) for view in observations},
    )

    assert report["quality_gate"]["passed"] is True
    assert report["quality_gate"]["valid_point_count"] == 6
    assert report["accepted_points_reference_m"]["nose_tip"] == pytest.approx(
        expected["nose_tip"], abs=1e-8
    )
    assert report["observed_profile_depth"]["nose_tip_minus_upper_lip_mm"] == pytest.approx(30.0)


def test_large_cross_view_semantic_error_is_not_converted_into_geometry():
    rig = _rig()
    _expected, observations = _semantic_observations(rig)
    corrupted = {
        view: {name: point.copy() for name, point in points.items()}
        for view, points in observations.items()
    }
    corrupted["right"]["nose_tip"] += np.array([150.0, 80.0])

    report = evaluate_semantic_profile_observations(
        corrupted,
        corrupted,
        rig,
        {view: (768, 1024, 3) for view in observations},
    )

    point = report["points"]["nose_tip"]
    assert point["passed"] is False
    assert "reprojection_error_exceeded" in point["issues"]
    assert "nose_tip" not in report["accepted_points_reference_m"]


def test_detector_semantic_disagreement_lowers_confidence_and_rejects_point():
    rig = _rig()
    _expected, observations = _semantic_observations(rig)
    face_alignment = {
        view: {name: point.copy() for name, point in points.items()}
        for view, points in observations.items()
    }
    for view in face_alignment:
        face_alignment[view]["chin"] += np.array([100.0, 0.0])

    report = evaluate_semantic_profile_observations(
        observations,
        face_alignment,
        rig,
        {view: (768, 1024, 3) for view in observations},
        observation_thresholds=ObservationThresholds(
            max_detector_disagreement_ratio=0.01,
            min_detector_confidence=0.5,
        ),
    )

    assert report["points"]["chin"]["passed"] is False
    assert "detector_disagreement_exceeded" in report["points"]["chin"]["issues"]
    assert report["quality_gate"]["passed"] is False

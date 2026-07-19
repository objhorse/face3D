from __future__ import annotations

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.profile_triangulation import (
    ProfileRig,
    TriangulationThresholds,
    profile_rig_from_payload,
    project_reference_point,
    triangulate_profile_point,
)


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


def _rig(baseline: float = 0.10) -> ProfileRig:
    cameras = {
        "left": _camera("camera1", "left", -baseline),
        "front": _camera("camera2", "front", 0.0),
        "right": _camera("camera3", "right", baseline),
    }
    return ProfileRig(
        cameras_by_view=cameras,
        reference_view="front",
        units="meters",
        calibration_path="synthetic",
        stereo_rms_px={"left": 0.1, "front": 0.0, "right": 0.1},
    )


def _observations(rig: ProfileRig, point: np.ndarray) -> dict[str, np.ndarray]:
    return {
        view: project_reference_point(point, view, rig)
        for view in ("left", "front", "right")
    }


def test_three_view_triangulation_recovers_known_point():
    rig = _rig()
    expected = np.array([0.018, -0.012, 0.55], dtype=np.float64)

    report = triangulate_profile_point(_observations(rig, expected), rig)

    assert report["passed"] is True
    assert np.asarray(report["point_reference_m"]) == pytest.approx(expected, abs=1e-8)
    assert report["reprojection_p90_px"] == pytest.approx(0.0, abs=1e-8)
    assert report["min_ray_angle_deg"] > 3.0


def test_triangulation_is_independent_of_observation_order():
    rig = _rig()
    expected = np.array([-0.011, 0.006, 0.48], dtype=np.float64)
    observations = _observations(rig, expected)

    forward = triangulate_profile_point(observations, rig)
    reverse = triangulate_profile_point(dict(reversed(list(observations.items()))), rig)

    assert reverse["passed"] is True
    assert reverse["point_reference_m"] == pytest.approx(
        forward["point_reference_m"], abs=1e-10
    )


def test_swapped_side_observations_are_rejected():
    rig = _rig()
    expected = np.array([0.032, 0.004, 0.52], dtype=np.float64)
    observations = _observations(rig, expected)
    observations["left"], observations["right"] = (
        observations["right"],
        observations["left"],
    )

    report = triangulate_profile_point(observations, rig)

    assert report["passed"] is False
    assert set(report["issues"]) & {
        "negative_or_invalid_depth",
        "reprojection_error_exceeded",
        "pairwise_3d_inconsistency",
    }


def test_low_ray_angle_is_rejected():
    rig = _rig(baseline=0.0001)
    point = np.array([0.01, 0.0, 0.8], dtype=np.float64)

    report = triangulate_profile_point(
        _observations(rig, point),
        rig,
        thresholds=TriangulationThresholds(min_ray_angle_deg=1.0),
    )

    assert report["passed"] is False
    assert "ray_angle_too_small" in report["issues"]


def test_profile_rig_rejects_bad_calibration_rms():
    cameras = {}
    for camera in _rig().cameras_by_view.values():
        cameras[camera.name] = {
            "view": camera.view,
            "image_size": list(camera.image_size),
            "K": camera.K.tolist(),
            "dist_coeffs": camera.dist.tolist(),
            "rig_to_camera": {
                "R": camera.R_rig_to_camera.tolist(),
                "t": camera.t_rig_to_camera.tolist(),
            },
            "stereo_rms": 42.0 if camera.view != "front" else 0.0,
        }
    payload = {
        "reference_camera": "camera2",
        "units": "meters",
        "view_aliases": {"camera1": "left", "camera2": "front", "camera3": "right"},
        "cameras": cameras,
    }
    with pytest.raises(ValueError, match="stereo RMS"):
        profile_rig_from_payload(
            payload,
            calibration_path="synthetic_bad_calibration.json",
            max_stereo_rms_px=10.0,
        )

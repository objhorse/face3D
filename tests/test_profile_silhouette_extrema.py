from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.observation_coordinates import (
    ObservationCoordinates,
    fundamental_matrix_work,
)
from src.geometry.profile_silhouette_extrema import (
    SilhouetteExtremumThresholds,
    _select_contour_candidate,
    build_profile_silhouette_extrema,
    canvas_points_to_original,
    original_points_to_canvas,
    refine_profile_extremum,
)
from src.geometry.profile_triangulation import ProfileRig, project_reference_point


def _camera(name: str, view: str, center_x: float) -> Camera:
    rotation = np.eye(3, dtype=np.float64)
    center = np.array([center_x, 0.0, 0.0], dtype=np.float64)
    return Camera(
        name=name,
        view=view,
        image_size=(640, 480),
        K=np.array(
            [[520.0, 0.0, 320.0], [0.0, 520.0, 240.0], [0.0, 0.0, 1.0]],
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


def _profile_mask(side: str) -> np.ndarray:
    mask = np.zeros((480, 640), dtype=np.uint8)
    left_profile = np.array(
        [
            [430, 120],
            [420, 180],
            [415, 231],
            [400, 270],
            [410, 321],
            [430, 380],
            [600, 410],
            [620, 100],
        ],
        dtype=np.int32,
    )
    points = (
        left_profile
        if side == "left"
        else np.column_stack((640 - left_profile[:, 0], left_profile[:, 1]))
    )
    cv2.fillPoly(mask, [points], 255)
    return mask


def _letterbox_mask(mask: np.ndarray, size: int) -> np.ndarray:
    resized_height = int(size * 3 / 4)
    resized = cv2.resize(
        mask,
        (size, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - resized_height) // 2
    canvas[y0 : y0 + resized_height] = resized
    return canvas


def test_letterbox_point_conversion_round_trips():
    points = np.array([[0.0, 0.0], [2448.0, 1836.0], [4895.0, 3671.0]])
    canvas = original_points_to_canvas(points, (4896, 3672), (1024, 1024))
    restored = canvas_points_to_original(canvas, (4896, 3672), (1024, 1024))

    assert restored == pytest.approx(points, abs=1e-8)
    assert canvas[0, 1] == pytest.approx(128.0)


def test_silhouette_extrema_recover_front_ray_depth():
    rig = _rig()
    nose = np.array([0.0, -0.01, 0.55], dtype=np.float64)
    chin = np.array([0.0, 0.09, 0.58], dtype=np.float64)
    masks = {
        "left": _profile_mask("left"),
        "right": _profile_mask("right"),
    }
    report = {
        "undistorted_mediapipe_px": {
            "front": {
                "nose_tip": project_reference_point(nose, "front", rig).tolist(),
                "chin": project_reference_point(chin, "front", rig).tolist(),
            }
        },
        "detectors": {
            side: {
                "nose_tip": {
                    "mediapipe_px": project_reference_point(
                        nose, side, rig
                    ).tolist()
                },
                "chin": {
                    "mediapipe_px": project_reference_point(
                        chin, side, rig
                    ).tolist()
                },
            }
            for side in ("left", "right")
        },
        "accepted_points_reference_m": {},
    }
    limits = SilhouetteExtremumThresholds(
        max_epipolar_distance_work_px=5.0,
        nose_prior_radius_work_px=90.0,
        chin_prior_radius_work_px=90.0,
        max_variant_spread_work_px=6.0,
        max_variant_depth_spread_m=0.030,
        max_cross_side_depth_delta_m=0.040,
    )

    result = build_profile_silhouette_extrema(
        report,
        masks,
        rig,
        thresholds=limits,
    )

    assert result["quality_gate"]["passed"] is True
    for name in ("nose_tip", "chin"):
        point = np.asarray(result["accepted_points_reference_m"][name])
        assert np.isfinite(point).all()
        assert 0.45 < point[2] < 0.75


def test_silhouette_extrema_reject_empty_masks():
    rig = _rig()
    point = np.array([0.0, 0.0, 0.55], dtype=np.float64)
    pixels = {
        view: project_reference_point(point, view, rig).tolist()
        for view in ("left", "front", "right")
    }
    report = {
        "undistorted_mediapipe_px": {
            "front": {"nose_tip": pixels["front"], "chin": pixels["front"]}
        },
        "detectors": {
            side: {
                "nose_tip": {"mediapipe_px": pixels[side]},
                "chin": {"mediapipe_px": pixels[side]},
            }
            for side in ("left", "right")
        },
        "accepted_points_reference_m": {},
    }

    result = build_profile_silhouette_extrema(
        report,
        {
            "left": np.zeros((480, 640), dtype=np.uint8),
            "right": np.zeros((480, 640), dtype=np.uint8),
        },
        rig,
    )

    assert result["quality_gate"]["passed"] is False
    assert set(result["quality_gate"]["missing_points"]) == {"nose_tip", "chin"}


def test_epipolar_threshold_filters_candidates_before_scoring():
    rig = _rig()
    front_camera = rig.cameras_by_view["front"]
    side_camera = rig.cameras_by_view["left"]
    front_anchor = np.array([320.0, 240.0], dtype=np.float64)
    front_coordinates = ObservationCoordinates.from_camera(
        front_camera,
        work_size=(640, 480),
        pixel_frame="undistorted",
    )
    side_coordinates = ObservationCoordinates.from_camera(
        side_camera,
        work_size=(640, 480),
        pixel_frame="distorted",
    )
    line = fundamental_matrix_work(
        front_camera,
        side_camera,
        front_coordinates,
        side_coordinates,
    ) @ np.append(front_anchor, 1.0)
    normal = line[:2] / np.linalg.norm(line[:2])
    tangent = np.array([-normal[1], normal[0]])
    point_on_line = -line[2] * line[:2] / np.dot(line[:2], line[:2])
    legal = point_on_line + 4.9 * normal + 19.0 * tangent
    illegal = point_on_line + 5.1 * normal
    contour = np.asarray(
        [
            legal,
            illegal,
            illegal + tangent,
            illegal + 2.0 * tangent,
            illegal + 3.0 * tangent,
            illegal + 4.0 * tangent,
            illegal + 5.0 * tangent,
        ]
    )
    thresholds = SilhouetteExtremumThresholds(
        max_epipolar_distance_work_px=5.0,
        nose_prior_radius_work_px=20.0,
    )

    result = _select_contour_candidate(
        contour,
        front_anchor,
        illegal,
        front_camera,
        side_camera,
        semantic_name="nose_tip",
        thresholds=thresholds,
    )

    assert result["passed"] is True
    assert result["selected_work_px"] == pytest.approx(legal, abs=1e-8)
    assert result["epipolar_distance_work_px"] == pytest.approx(4.9, abs=1e-8)


def test_profile_mask_variants_are_independent_of_letterbox_resolution():
    rig = _rig()
    thresholds = SilhouetteExtremumThresholds(
        max_epipolar_distance_work_px=6.0,
        nose_prior_radius_work_px=100.0,
        mask_variant_offsets_work_px=(-2, 0, 2),
        max_variant_spread_work_px=8.0,
        max_variant_depth_spread_m=0.050,
    )
    mask = _profile_mask("left")
    selected = []

    for size in (160, 640, 1024):
        result = refine_profile_extremum(
            np.array([320.0, 270.0]),
            np.array([400.0, 270.0]),
            _letterbox_mask(mask, size),
            rig,
            "left",
            "nose_tip",
            thresholds=thresholds,
        )
        selected.append(
            np.asarray(
                result["selected_center"]["selection"]["selected_work_px"],
                dtype=np.float64,
            )
        )

    assert np.max(np.ptp(np.asarray(selected), axis=0)) <= 4.1

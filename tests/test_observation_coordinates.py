from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.observation_coordinates import (
    ObservationCoordinates,
    canvas_points_to_original,
    intrinsics_to_letterbox_canvas,
    normalize_image_to_work,
    normalize_mask_to_work,
    original_points_to_canvas,
    original_points_to_work,
    work_points_to_original,
)


def test_intrinsics_map_from_rectangular_work_frame_to_square_letterbox():
    source_size = (640, 480)
    canvas_shape = (1024, 1024)
    intrinsics = np.array(
        [
            [445.0, 0.0, 303.0],
            [0.0, 444.0, 211.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )

    mapped = intrinsics_to_letterbox_canvas(
        intrinsics,
        source_size,
        canvas_shape,
    )

    assert mapped == pytest.approx(
        np.array(
            [
                [712.0, 0.0, 484.8],
                [0.0, 710.4, 465.6],
                [0.0, 0.0, 1.0],
            ]
        )
    )


def _camera(
    *,
    image_size: tuple[int, int] = (4896, 3672),
    dist: np.ndarray | None = None,
) -> Camera:
    width, height = image_size
    return Camera(
        name="camera2",
        view="front",
        image_size=image_size,
        K=np.array(
            [
                [4100.0, 0.0, width / 2.0],
                [0.0, 4090.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        dist=(
            np.array([0.08, -0.04, 0.001, -0.002, 0.01], dtype=np.float64)
            if dist is None
            else np.asarray(dist, dtype=np.float64)
        ),
        R_rig_to_camera=np.eye(3, dtype=np.float64),
        t_rig_to_camera=np.zeros(3, dtype=np.float64),
    )


def test_distorted_original_points_round_trip_and_letterbox_mask_normalizes():
    camera = _camera()
    coordinates = ObservationCoordinates.from_camera(
        camera,
        work_size=(640, 480),
        pixel_frame="distorted",
    )
    points = np.array(
        [[900.0, 720.0], [2448.0, 1836.0], [4010.0, 2920.0]],
        dtype=np.float64,
    )

    work = original_points_to_work(points, coordinates)
    restored = work_points_to_original(
        work,
        coordinates,
        target_pixel_frame="distorted",
    )

    assert restored == pytest.approx(points, abs=1e-5)

    canvas = np.zeros((1024, 1024), dtype=np.uint8)
    polygon_original = np.array(
        [[1600.0, 1200.0], [3300.0, 1200.0], [3300.0, 2600.0], [1600.0, 2600.0]]
    )
    polygon_canvas = np.rint(
        original_points_to_canvas(
            polygon_original,
            camera.image_size,
            canvas.shape,
        )
    ).astype(np.int32)
    cv2.fillPoly(canvas, [polygon_canvas], 255)

    mask_work = normalize_mask_to_work(canvas, coordinates, name="nose mask")
    image_work = normalize_image_to_work(
        np.zeros((3672, 4896, 3), dtype=np.uint8),
        coordinates,
    )

    assert mask_work.shape == (480, 640)
    assert image_work.shape == (480, 640, 3)
    assert mask_work.dtype == np.uint8
    assert set(np.unique(mask_work)) <= {0, 1}
    assert mask_work[250, 320] == 1
    restored_canvas = canvas_points_to_original(
        polygon_canvas,
        camera.image_size,
        canvas.shape,
    )
    assert restored_canvas == pytest.approx(polygon_original, abs=5.0)


def test_undistorted_contract_never_applies_a_second_undistortion(monkeypatch):
    camera = _camera(dist=np.zeros(5, dtype=np.float64))
    new_intrinsics = camera.K.copy()
    new_intrinsics[0, 0] = 4050.0
    coordinates = ObservationCoordinates(
        original_size=camera.image_size,
        work_size=(640, 480),
        K=new_intrinsics,
        dist=np.zeros(5, dtype=np.float64),
        pixel_frame="undistorted",
    )
    points = np.array([[0.0, 0.0], [2448.0, 1836.0], [4895.0, 3671.0]])
    image = np.zeros((3672, 4896, 3), dtype=np.uint8)

    def fail(*_args, **_kwargs):
        raise AssertionError("undistortion must not run for an undistorted frame")

    monkeypatch.setattr(cv2, "undistort", fail)
    monkeypatch.setattr(cv2, "undistortPoints", fail)

    work = original_points_to_work(points, coordinates)
    frame_work = normalize_image_to_work(image, coordinates)

    expected = points * np.array([640.0 / 4896.0, 480.0 / 3672.0])
    assert work == pytest.approx(expected, abs=1e-10)
    assert work_points_to_original(work, coordinates) == pytest.approx(
        points,
        abs=1e-10,
    )
    assert frame_work.shape == (480, 640, 3)


def test_undistorted_contract_rejects_distorted_reverse_conversion():
    camera = _camera(dist=np.zeros(5, dtype=np.float64))
    coordinates = ObservationCoordinates(
        original_size=camera.image_size,
        work_size=(640, 480),
        K=camera.K,
        dist=np.zeros(5, dtype=np.float64),
        pixel_frame="undistorted",
    )

    assert coordinates.metadata()["distorted_reverse_available"] is False
    with pytest.raises(ValueError, match="distorted.*unavailable"):
        work_points_to_original(
            np.array([[320.0, 240.0]], dtype=np.float64),
            coordinates,
            target_pixel_frame="distorted",
        )


def test_coordinate_contract_owns_read_only_calibration_copies():
    camera = _camera()
    coordinates = ObservationCoordinates.from_camera(
        camera,
        work_size=(640, 480),
        pixel_frame="distorted",
    )
    original_fx = float(coordinates.K[0, 0])

    camera.K[0, 0] = 1.0

    assert coordinates.K[0, 0] == original_fx
    with pytest.raises(ValueError):
        coordinates.K[0, 0] = 2.0
    with pytest.raises(ValueError):
        coordinates.dist[0] = 0.0

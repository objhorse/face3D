import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.appearance.projective_sampling import (
    DEFAULT_POSITIVE_DEPTH_EPSILON,
    ProjectionCoordinates,
    ProjectionSample,
    assert_strict_sampling_coordinates,
    compare_sampling_coordinates,
    project_points_strict,
    render_camera_depth,
    sample_projected_attributes,
)
from src.coordinates import project_texture_points_to_image


def test_project_points_strict_applies_world_to_camera_perspective_projection():
    points = np.array(
        [
            [0.0, 0.0, 2.0],
            [1.0, 2.0, 4.0],
            [-1.0, 1.0, 1.0],
        ]
    )
    intrinsics = np.array(
        [
            [100.0, 0.0, 320.0],
            [0.0, 200.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )
    translation = np.array([1.0, -2.0, 1.0])

    result = project_points_strict(points, intrinsics, rotation, translation)

    expected_camera = points @ rotation.T + translation
    expected_pixels = np.column_stack(
        (
            100.0 * expected_camera[:, 0] / expected_camera[:, 2] + 320.0,
            200.0 * expected_camera[:, 1] / expected_camera[:, 2] + 240.0,
        )
    )
    assert isinstance(result, ProjectionCoordinates)
    np.testing.assert_allclose(result.camera_points, expected_camera)
    np.testing.assert_allclose(result.depth, expected_camera[:, 2])
    np.testing.assert_allclose(result.pixel_xy, expected_pixels)
    np.testing.assert_array_equal(result.front_facing, [True, True, True])


def test_projection_coordinates_are_immutable_and_own_read_only_arrays():
    result = project_points_strict(
        np.array([[0.0, 0.0, 1.0]]),
        np.eye(3),
        np.eye(3),
        np.zeros(3),
    )

    with pytest.raises(FrozenInstanceError):
        result.depth = np.array([2.0])
    with pytest.raises(ValueError):
        result.pixel_xy[0, 0] = 10.0


def test_projected_attributes_share_one_pixel_coordinate_for_all_rasters():
    coordinates = ProjectionCoordinates(
        camera_points=np.array([[0.0, 0.0, 2.0], [0.0, 0.0, 2.0]]),
        depth=np.array([2.0, 2.0]),
        pixel_xy=np.array([[1.25, 1.5], [8.0, 8.0]]),
        front_facing=np.array([True, True]),
    )
    image = np.zeros((4, 4, 3), dtype=np.float32)
    image[1, 1] = [10.0, 20.0, 30.0]
    image[1, 2] = [30.0, 40.0, 50.0]
    image[2, 1] = [50.0, 60.0, 70.0]
    image[2, 2] = [70.0, 80.0, 90.0]
    mask = np.zeros((4, 4), dtype=np.uint8)
    mask[1, 1] = 255
    depth = np.full((4, 4), 9.0, dtype=np.float32)
    depth[1, 1] = 2.25
    semantic = np.zeros((4, 4), dtype=np.uint8)
    semantic[1, 1] = 7

    sample = sample_projected_attributes(
        coordinates,
        image,
        mask=mask,
        depth_map=depth,
        semantic_map=semantic,
    )

    assert isinstance(sample, ProjectionSample)
    np.testing.assert_array_equal(sample.in_bounds, [True, False])
    np.testing.assert_allclose(sample.rgb[0], [35.0, 45.0, 55.0])
    assert sample.mask.tolist() == [True, False]
    assert sample.depth[0] == pytest.approx(2.25)
    assert sample.semantic[0] == pytest.approx(7.0)
    with pytest.raises(ValueError):
        sample.rgb[0, 0] = 0.0


def test_projected_attributes_require_aligned_raster_shapes():
    coordinates = project_points_strict(
        np.array([[0.0, 0.0, 2.0]]),
        np.eye(3),
        np.eye(3),
        np.zeros(3),
    )

    with pytest.raises(ValueError, match="mask.*match"):
        sample_projected_attributes(
            coordinates,
            np.zeros((4, 4, 3), dtype=np.uint8),
            mask=np.zeros((3, 4), dtype=np.uint8),
        )


def test_render_camera_depth_uses_the_strict_projection_pixels():
    vertices = np.array(
        [[-0.5, -0.5, 2.0], [0.0, 0.5, 2.0], [0.5, -0.5, 2.0]],
        dtype=np.float32,
    )
    depth = render_camera_depth(
        vertices,
        np.array([[0, 1, 2]], dtype=np.int32),
        np.array([[8.0, 0.0, 8.0], [0.0, 8.0, 8.0], [0.0, 0.0, 1.0]]),
        np.eye(3),
        np.zeros(3),
        (16, 16),
    )

    assert depth.shape == (16, 16)
    assert np.isfinite(depth).any()
    np.testing.assert_allclose(depth[np.isfinite(depth)], 2.0)


def test_projection_coordinates_validate_depth_and_binary_front_facing():
    camera_points = np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]])
    pixels = np.array([[10.0, 20.0], [0.0, 0.0]])

    result = ProjectionCoordinates(
        camera_points,
        np.array([2.0, -1.0]),
        pixels,
        np.array([1.0, 0.0]),
    )

    assert result.front_facing.dtype == np.bool_
    with pytest.raises(ValueError, match="depth.*camera_points"):
        ProjectionCoordinates(
            camera_points,
            np.array([3.0, -1.0]),
            pixels,
            np.array([True, False]),
        )


@pytest.mark.parametrize(
    "front_facing",
    [
        np.array([np.nan, 0.0]),
        np.array([2, 0]),
        np.array([-1, 0]),
    ],
)
def test_projection_coordinates_reject_nonbinary_front_facing(front_facing):
    with pytest.raises(ValueError, match="front_facing.*bool.*0/1"):
        ProjectionCoordinates(
            np.array([[0.0, 0.0, 2.0], [0.0, 0.0, -1.0]]),
            np.array([2.0, -1.0]),
            np.array([[10.0, 20.0], [0.0, 0.0]]),
            front_facing,
        )


@pytest.mark.parametrize(
    ("points", "intrinsics", "rotation", "translation"),
    [
        (np.zeros((2, 2)), np.eye(3), np.eye(3), np.zeros(3)),
        (np.zeros((2, 3)), np.eye(4), np.eye(3), np.zeros(3)),
        (np.zeros((2, 3)), np.eye(3), np.eye(4), np.zeros(3)),
        (np.zeros((2, 3)), np.eye(3), np.eye(3), np.zeros((3, 1))),
    ],
)
def test_project_points_strict_rejects_invalid_shapes(
    points, intrinsics, rotation, translation
):
    with pytest.raises(ValueError, match="shape"):
        project_points_strict(points, intrinsics, rotation, translation)


@pytest.mark.parametrize("argument", ["points", "intrinsics", "rotation", "translation"])
def test_project_points_strict_rejects_nonfinite_inputs(argument):
    inputs = {
        "points": np.zeros((2, 3)),
        "intrinsics": np.eye(3),
        "rotation": np.eye(3),
        "translation": np.zeros(3),
    }
    inputs[argument] = inputs[argument].copy()
    inputs[argument].flat[0] = np.nan

    with pytest.raises(ValueError, match="finite"):
        project_points_strict(
            inputs["points"],
            inputs["intrinsics"],
            inputs["rotation"],
            inputs["translation"],
        )


def test_project_points_strict_marks_zero_negative_and_epsilon_depth_invalid():
    result = project_points_strict(
        np.array(
            [
                [1.0, 2.0, 2.0],
                [1.0, 2.0, 0.0],
                [1.0, 2.0, -1.0],
                [1.0, 2.0, 5e-5],
            ]
        ),
        np.eye(3),
        np.eye(3),
        np.zeros(3),
    )

    np.testing.assert_array_equal(result.front_facing, [True, False, False, False])
    np.testing.assert_allclose(result.pixel_xy[0], [0.5, 1.0])
    np.testing.assert_array_equal(result.pixel_xy[1:], np.zeros((3, 2)))


def test_legacy_projection_adapter_matches_strict_contract_for_real_camera_values():
    points = np.array(
        [
            [0.018, -0.032, 0.241],
            [-0.067, 0.014, 0.315],
            [0.001, 0.002, 5e-5],
        ],
        dtype=np.float64,
    )
    intrinsics = np.array(
        [[1843.72, 0.0, 960.34], [0.0, 1841.91, 604.82], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    angle = np.deg2rad(27.5)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    translation = np.array([0.021, -0.008, 0.613], dtype=np.float64)

    strict = project_points_strict(points, intrinsics, rotation, translation)
    camera_points, depth, pixels, front = project_texture_points_to_image(
        points, intrinsics, rotation, translation
    )

    assert DEFAULT_POSITIVE_DEPTH_EPSILON == 1e-4
    assert strict.camera_points.dtype == np.float32
    assert strict.depth.dtype == np.float32
    assert strict.pixel_xy.dtype == np.float32
    np.testing.assert_array_equal(camera_points, strict.camera_points)
    np.testing.assert_array_equal(depth, strict.depth)
    np.testing.assert_array_equal(pixels, strict.pixel_xy)
    np.testing.assert_array_equal(front, strict.front_facing)
    assert_strict_sampling_coordinates(pixels, strict.pixel_xy, front)


def test_projection_contract_agrees_at_positive_depth_threshold():
    points = np.array([[0.0, 0.0, 5e-5], [0.0, 0.0, 2e-4]])
    strict = project_points_strict(points, np.eye(3), np.eye(3), np.zeros(3))
    camera_points, depth, pixels, front = project_texture_points_to_image(
        points, np.eye(3), np.eye(3), np.zeros(3)
    )

    np.testing.assert_array_equal(front, [False, True])
    np.testing.assert_array_equal(camera_points, strict.camera_points)
    np.testing.assert_array_equal(depth, strict.depth)
    np.testing.assert_array_equal(pixels, strict.pixel_xy)


def test_compare_sampling_coordinates_reports_json_serializable_percentiles():
    projected = np.zeros((5, 2), dtype=np.float64)
    sampled = np.array(
        [
            [0.0, 0.0],
            [1.0, 0.0],
            [3.0, 4.0],
            [13.0, 0.0],
            [20.0, 0.0],
        ]
    )

    metrics = compare_sampling_coordinates(projected, sampled)

    assert metrics == {
        "count": 5,
        "mean_displacement_px": pytest.approx(7.8),
        "p50_displacement_px": pytest.approx(5.0),
        "p90_displacement_px": pytest.approx(17.2),
        "p95_displacement_px": pytest.approx(18.6),
        "max_displacement_px": pytest.approx(20.0),
        "exact_coordinate_ratio": pytest.approx(0.2),
        "over_1px_ratio": pytest.approx(0.6),
        "over_5px_ratio": pytest.approx(0.4),
        "over_12px_ratio": pytest.approx(0.4),
    }
    json.dumps(metrics, allow_nan=False)


def test_compare_sampling_coordinates_respects_valid_mask():
    projected = np.zeros((4, 2))
    sampled = np.array([[0.0, 0.0], [2.0, 0.0], [6.0, 0.0], [15.0, 0.0]])

    metrics = compare_sampling_coordinates(
        projected,
        sampled,
        valid_mask=np.array([True, False, True, False]),
    )

    assert metrics["count"] == 2
    assert metrics["mean_displacement_px"] == pytest.approx(3.0)
    assert metrics["p50_displacement_px"] == pytest.approx(3.0)
    assert metrics["over_5px_ratio"] == pytest.approx(0.5)


def test_compare_sampling_coordinates_empty_mask_is_explicit_and_json_serializable():
    metrics = compare_sampling_coordinates(
        np.zeros((2, 2)),
        np.ones((2, 2)),
        valid_mask=np.array([False, False]),
    )

    assert metrics == {
        "count": 0,
        "mean_displacement_px": None,
        "p50_displacement_px": None,
        "p90_displacement_px": None,
        "p95_displacement_px": None,
        "max_displacement_px": None,
        "exact_coordinate_ratio": 0.0,
        "over_1px_ratio": 0.0,
        "over_5px_ratio": 0.0,
        "over_12px_ratio": 0.0,
    }
    json.dumps(metrics, allow_nan=False)


def test_compare_sampling_coordinates_rejects_nonfinite_displacement_overflow():
    with pytest.raises(ValueError, match="displacement.*finite"):
        compare_sampling_coordinates(
            np.array([[1e308, 1e308]]),
            np.array([[-1e308, -1e308]]),
        )


def test_assert_strict_sampling_coordinates_accepts_identical_coordinates():
    coordinates = np.array([[1.0, 2.0], [3.0, 4.0]])

    metrics = assert_strict_sampling_coordinates(coordinates, coordinates.copy())

    assert metrics["count"] == 2
    assert metrics["exact_coordinate_ratio"] == 1.0


def test_assert_strict_sampling_coordinates_rejects_warped_coordinates():
    projected = np.array([[1.0, 2.0], [3.0, 4.0]])
    sampled = projected.copy()
    sampled[1, 0] += 0.25

    with pytest.raises(ValueError, match="strict projective sampling"):
        assert_strict_sampling_coordinates(projected, sampled)


def test_coordinate_comparison_rejects_bad_shapes_masks_and_nonfinite_values():
    with pytest.raises(ValueError, match="matching.*shape"):
        compare_sampling_coordinates(np.zeros((2, 2)), np.zeros((3, 2)))
    with pytest.raises(ValueError, match="mask.*shape"):
        compare_sampling_coordinates(
            np.zeros((2, 2)), np.zeros((2, 2)), np.array([True])
        )
    with pytest.raises(ValueError, match="finite"):
        compare_sampling_coordinates(
            np.array([[np.nan, 0.0]]), np.zeros((1, 2))
        )


@pytest.mark.parametrize(
    "valid_mask",
    [
        np.array([True, np.nan]),
        np.array([1, 2]),
        np.array([0, -1]),
    ],
)
def test_coordinate_comparison_rejects_nonbinary_valid_mask(valid_mask):
    with pytest.raises(ValueError, match="valid mask.*bool.*0/1"):
        compare_sampling_coordinates(
            np.zeros((2, 2)),
            np.zeros((2, 2)),
            valid_mask=valid_mask,
        )


def test_coordinate_comparison_accepts_finite_zero_one_valid_mask():
    metrics = compare_sampling_coordinates(
        np.zeros((2, 2)),
        np.array([[0.0, 0.0], [9.0, 0.0]]),
        valid_mask=np.array([1.0, 0.0]),
    )

    assert metrics["count"] == 1
    assert metrics["exact_coordinate_ratio"] == 1.0

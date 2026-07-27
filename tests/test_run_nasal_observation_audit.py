from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from run_nasal_observation_audit import (
    assert_file_tree_unchanged,
    build_undistorted_observation_rig,
    file_tree_hashes,
    make_nasal_audit_config,
    project_flame_points_through_front_fit,
    require_parser_nose_mask,
    validate_separate_output,
)
from src.cross_view_geometry import Camera
from src.geometry.observation_coordinates import ObservationCoordinates
from src.geometry.profile_triangulation import ProfileRig


def _camera(
    name: str,
    view: str,
    *,
    rotation: np.ndarray | None = None,
    translation: np.ndarray | None = None,
) -> Camera:
    return Camera(
        name=name,
        view=view,
        image_size=(160, 120),
        K=np.array(
            [[140.0, 0.0, 80.0], [0.0, 145.0, 60.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist=np.array([0.1, -0.03, 0.001, 0.0, 0.0], dtype=np.float64),
        R_rig_to_camera=(
            np.eye(3, dtype=np.float64)
            if rotation is None
            else np.asarray(rotation, dtype=np.float64)
        ),
        t_rig_to_camera=(
            np.zeros(3, dtype=np.float64)
            if translation is None
            else np.asarray(translation, dtype=np.float64)
        ),
    )


def _rig() -> ProfileRig:
    angle = np.deg2rad(7.0)
    side_rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )
    front_angle = np.deg2rad(-4.0)
    front_rotation = np.array(
        [
            [np.cos(front_angle), -np.sin(front_angle), 0.0],
            [np.sin(front_angle), np.cos(front_angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return ProfileRig(
        cameras_by_view={
            "left": _camera(
                "camera1",
                "left",
                rotation=side_rotation,
                translation=np.array([0.08, 0.01, 0.0]),
            ),
            "front": _camera(
                "camera2",
                "front",
                rotation=front_rotation,
                translation=np.array([0.02, -0.01, 0.03]),
            ),
            "right": _camera(
                "camera3",
                "right",
                rotation=side_rotation.T,
                translation=np.array([-0.08, 0.0, 0.01]),
            ),
        },
        reference_view="front",
        units="meters",
        calibration_path="synthetic-rig.json",
        stereo_rms_px={"left": 0.2, "front": 0.0, "right": 0.2},
    )


def test_undistorted_new_k_and_zero_dist_contract():
    rig = _rig()
    new_intrinsics = {
        view: camera.K + np.diag([3.0, 4.0, 0.0])
        for view, camera in rig.cameras_by_view.items()
    }

    observation_rig = build_undistorted_observation_rig(
        rig,
        new_intrinsics,
    )

    for view, camera in observation_rig.cameras_by_view.items():
        assert camera.K == pytest.approx(new_intrinsics[view])
        assert camera.dist == pytest.approx(np.zeros_like(camera.dist))
        assert camera.R_rig_to_camera == pytest.approx(
            rig.cameras_by_view[view].R_rig_to_camera
        )
        coordinates = ObservationCoordinates.from_camera(
            camera,
            work_size=(64, 48),
            pixel_frame="undistorted",
        )
        assert coordinates.pixel_frame == "undistorted"
        assert coordinates.dist == pytest.approx(np.zeros_like(camera.dist))


def test_audit_config_scales_one_fixed_roi_rule_with_work_resolution():
    full = make_nasal_audit_config((640, 480))
    half = make_nasal_audit_config((320, 240))

    assert full.max_side_prior_distance_px == pytest.approx(80.0)
    assert full.max_epipolar_distance_px == pytest.approx(8.0)
    assert half.max_side_prior_distance_px == pytest.approx(40.0)
    assert half.max_epipolar_distance_px == pytest.approx(4.0)
    assert full.mask_perturbation_px == 2
    assert half.mask_perturbation_px == 1


def test_baseline_flame_front_camera_rig_side_projection_matches_manual_transform():
    rig = _rig()
    angle = np.deg2rad(-11.0)
    fit_rotation = np.array(
        [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    fit_translation = np.array([0.01, -0.02, 0.62], dtype=np.float64)
    points_flame = np.array(
        [[0.01, 0.02, 0.03], [-0.03, 0.01, 0.02]],
        dtype=np.float64,
    )

    projected = project_flame_points_through_front_fit(
        points_flame,
        fit_rotation,
        fit_translation,
        rig,
    )

    front_camera = rig.cameras_by_view["front"]
    points_front = points_flame @ fit_rotation.T + fit_translation
    points_rig = (
        points_front - front_camera.t_rig_to_camera
    ) @ front_camera.R_rig_to_camera
    for view, camera in rig.cameras_by_view.items():
        points_camera = (
            points_rig @ camera.R_rig_to_camera.T
            + camera.t_rig_to_camera
        )
        homogeneous = points_camera @ camera.K.T
        expected = homogeneous[:, :2] / homogeneous[:, 2:3]
        assert projected[view] == pytest.approx(expected)


def test_parser_nose_mask_missing_fails_explicitly():
    valid_mask = np.ones((16, 16), dtype=np.uint8) * 255
    preprocessed = {
        view: {
            "parser_labels": np.ones((16, 16), dtype=np.uint8),
            "nose_mask": valid_mask.copy(),
            "face_mask": valid_mask.copy(),
        }
        for view in ("left", "front", "right")
    }

    missing_parser = {
        view: dict(values)
        for view, values in preprocessed.items()
    }
    missing_parser["right"]["parser_labels"] = None
    with pytest.raises(RuntimeError, match="parser.*right"):
        require_parser_nose_mask(missing_parser)

    empty_nose = {
        view: dict(values)
        for view, values in preprocessed.items()
    }
    empty_nose["front"]["nose_mask"] = np.zeros((16, 16), dtype=np.uint8)
    with pytest.raises(RuntimeError, match="front nose_mask.*label 10"):
        require_parser_nose_mask(empty_nose)

    absent_nose = {
        view: dict(values)
        for view, values in preprocessed.items()
    }
    absent_nose["front"]["nose_mask"] = None
    with pytest.raises(RuntimeError, match="front nose_mask.*label 10"):
        require_parser_nose_mask(absent_nose)


def test_file_tree_guard_detects_source_changes_and_ignores_external_output(
    tmp_path,
):
    source = tmp_path / "source"
    output = tmp_path / "output"
    source.mkdir()
    output.mkdir()
    source_file = source / "stable_fit_meta.json"
    source_file.write_text("baseline", encoding="utf-8")
    before = file_tree_hashes(source)

    (output / "nasal_observations.json").write_text("{}", encoding="utf-8")
    assert_file_tree_unchanged(source, before)

    source_file.write_text("changed", encoding="utf-8")
    with pytest.raises(RuntimeError, match="source files changed"):
        assert_file_tree_unchanged(source, before)


def test_output_must_not_overlap_source_or_capture_directories(tmp_path):
    source = tmp_path / "source"
    captures = tmp_path / "captures"
    source.mkdir()
    captures.mkdir()

    validate_separate_output(tmp_path / "audit", source, captures)
    with pytest.raises(ValueError, match="source-output"):
        validate_separate_output(source / "audit", source, captures)
    with pytest.raises(ValueError, match="capture-dir"):
        validate_separate_output(captures, source, captures)

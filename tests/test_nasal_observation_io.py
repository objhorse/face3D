from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.observation_coordinates import ObservationCoordinates
from src.reports.nasal_observation_io import load_nasal_observation_bundle
from src.reports.nasal_observation_report import write_nasal_observation_data


def _camera(name: str, view: str, tx: float) -> Camera:
    return Camera(
        name=name,
        view=view,
        image_size=(12, 8),
        K=np.array([[20.0, 0.0, 6.0], [0.0, 20.0, 4.0], [0.0, 0.0, 1.0]]),
        dist=np.zeros(5),
        R_rig_to_camera=np.eye(3),
        t_rig_to_camera=np.array([tx, 0.0, 0.0]),
    )


def _observation(semantic_view: str, camera: Camera) -> NasalViewObservation:
    work_size = (6, 4)
    names = (
        ("subject-left-alar", "subject-right-alar")
        if semantic_view == "front"
        else ("nasal-profile",)
    )
    curves = {
        name: np.array([[1.0, 1.0], [2.0, 2.0], [3.0, 2.5]])
        for name in names
    }
    boundary = np.zeros((4, 6), dtype=np.uint8)
    boundary[1, 1:4] = 1
    distance = np.ones((4, 6), dtype=np.float32)
    coordinates = ObservationCoordinates.from_camera(
        camera,
        work_size=work_size,
        pixel_frame="undistorted",
    )
    coordinate_metadata = coordinates.metadata()
    coordinate_metadata.update(
        {
            "mask_pixel_layout": "original",
            "distance_field": "unsigned_truncated_precise_euclidean_work_px",
            "aggregate_distance_field_usage": "display_only",
            "mask_canvas_shape_hw": [8, 12],
            "color_space": "RGB",
        }
    )
    if semantic_view != "front":
        coordinate_metadata.update(
            {
                "epipolar_lines_work": [[1.0, 0.0, -2.0]] * 3,
                "epipolar_anchor_names": [
                    "upper_tip",
                    "tip_apex",
                    "lower_tip",
                ],
                "max_epipolar_distance_px": 4.0,
                "candidate_count": 1,
                "selected_anchor_epipolar_errors_px": [0.1, 0.2, 0.1],
                "selected_anchor_prior_errors_px": [0.2, 0.1, 0.2],
                "side_prior_roi_work": [0.0, 0.0, 6.0, 4.0],
                "effective_profile_roi_work": [0.0, 0.0, 6.0, 4.0],
                "max_side_prior_distance_px": 8.0,
                "selection_score": 0.25,
                "used_side_prior": True,
            }
        )
    camera_metadata = {
        "camera_name": camera.name,
        "camera_view": camera.view,
        "subject_relative_view": semantic_view,
        "image_size_wh": list(camera.image_size),
        "intrinsics": camera.K.tolist(),
        "distortion_coefficients": camera.dist.tolist(),
        "rig_to_camera_rotation": camera.R_rig_to_camera.tolist(),
        "rig_to_camera_translation": camera.t_rig_to_camera.tolist(),
    }
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=camera.image_size,
        mask_canvas_shape=(8, 12),
        work_size=work_size,
        roi_work_xyxy=(0.0, 0.0, 6.0, 4.0),
        boundaries_work=curves,
        boundary=boundary,
        distance_fields={name: distance for name in names},
        distance_field=distance,
        confidence=np.full((4, 6), 0.75, dtype=np.float32),
        variant_boundaries_work={"base": curves},
        variant_boundaries={"base": boundary},
        anchors_work={"tip": np.array([2.0, 1.5])},
        camera_metadata=camera_metadata,
        coordinate_metadata=coordinate_metadata,
    )


def _write_bundle(root: Path) -> Path:
    bundle = NasalObservationBundle(
        front=_observation("front", _camera("camera2", "front", 0.0)),
        subject_left=_observation(
            "subject-left",
            _camera("camera1", "left", -0.1),
        ),
        subject_right=_observation(
            "subject-right",
            _camera("camera3", "right", 0.1),
        ),
    )
    metadata = root / "nasal_observations.json"
    write_nasal_observation_data(
        bundle,
        metadata,
        root / "nasal_observation_fields.npz",
        metadata={"dataset": "fixture"},
    )
    return metadata


def test_loader_roundtrips_audit_bundle(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)

    restored = load_nasal_observation_bundle(path)

    assert tuple(restored.by_view) == (
        "front",
        "subject-left",
        "subject-right",
    )
    assert restored.camera_name_by_view == {
        "front": "camera2",
        "subject-left": "camera1",
        "subject-right": "camera3",
    }
    assert np.array_equal(
        restored.subject_right.distance_fields["nasal-profile"],
        np.ones((4, 6), dtype=np.float32),
    )
    assert not restored.front.confidence.flags.writeable


def test_loader_rejects_missing_camera_calibration(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    del payload["views"]["front"]["camera"]["rig_to_camera_rotation"]
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="rig_to_camera_rotation"):
        load_nasal_observation_bundle(path)


def test_loader_rejects_invalid_camera_intrinsics(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["views"]["front"]["camera"]["intrinsics"][0][0] = 0.0
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="focal lengths"):
        load_nasal_observation_bundle(path)


@pytest.mark.parametrize(
    "container_path",
    [
        (),
        ("views", "front"),
        ("views", "front", "camera"),
        ("views", "front", "coordinate_metadata"),
        ("views", "front", "fields"),
    ],
)
def test_loader_rejects_unexpected_schema_keys(
    tmp_path: Path,
    container_path: tuple[str, ...],
) -> None:
    path = _write_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    container = payload
    for name in container_path:
        container = container[name]
    container["unexpected_payload"] = "forbidden"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="unexpected_payload"):
        load_nasal_observation_bundle(path)


@pytest.mark.parametrize("illegal_value", [float("nan"), float("inf")])
def test_loader_rejects_nonfinite_nested_metadata(
    tmp_path: Path,
    illegal_value: float,
) -> None:
    path = _write_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["metadata"]["untrusted"] = {
        "nested": [1.0, {"illegal": illegal_value}]
    }
    path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

    with pytest.raises(ValueError, match="non-finite"):
        load_nasal_observation_bundle(path)


@pytest.mark.parametrize("field_path", ["../escape.npz", "nested/../../escape.npz"])
def test_loader_rejects_fields_path_traversal(
    tmp_path: Path,
    field_path: str,
) -> None:
    path = _write_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["fields_npz"] = field_path
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="escapes"):
        load_nasal_observation_bundle(path)


def test_loader_rejects_object_npz_without_pickle(tmp_path: Path) -> None:
    path = _write_bundle(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    field_name = payload["views"]["front"]["fields"]["boundary"]
    fields_path = tmp_path / payload["fields_npz"]
    with np.load(fields_path, allow_pickle=False) as archive:
        fields = {name: np.array(archive[name], copy=True) for name in archive.files}
    fields[field_name] = np.array([{"forbidden": True}], dtype=object)
    np.savez_compressed(fields_path, **fields)

    with pytest.raises(ValueError, match="pickle|object"):
        load_nasal_observation_bundle(path)

from __future__ import annotations

import json
import re
from pathlib import Path

import cv2
import numpy as np

from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalViewObservation,
)
from src.geometry.observation_coordinates import ObservationCoordinates
from src.reports.nasal_observation_report import (
    write_nasal_observation_data,
    write_nasal_observation_report,
)


ANCHOR_NAMES = (
    "upper_tip",
    "tip_apex",
    "lower_tip",
    "alar_transition",
)


def _camera(name: str, view: str) -> Camera:
    return Camera(
        name=name,
        view=view,
        image_size=(32, 24),
        K=np.array(
            [[30.0, 0.0, 16.0], [0.0, 30.0, 12.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=np.eye(3, dtype=np.float64),
        t_rig_to_camera=np.zeros(3, dtype=np.float64),
    )


def _curve(x: float) -> np.ndarray:
    return np.column_stack(
        (
            np.full(8, x, dtype=np.float64),
            np.linspace(7.0, 16.0, 8, dtype=np.float64),
        )
    )


def _raster(curves: dict[str, np.ndarray]) -> np.ndarray:
    result = np.zeros((24, 32), dtype=np.uint8)
    for curve in curves.values():
        cv2.polylines(
            result,
            [np.rint(curve).astype(np.int32).reshape(-1, 1, 2)],
            False,
            1,
            1,
        )
    return result.astype(bool)


def _distance(boundary: np.ndarray) -> np.ndarray:
    return cv2.distanceTransform(
        np.asarray(~boundary, dtype=np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    ).astype(np.float32)


def _observation(
    semantic_view: str,
    camera: Camera,
) -> NasalViewObservation:
    if semantic_view == "front":
        curves = {
            "subject-left-alar": _curve(20.0),
            "subject-right-alar": _curve(12.0),
        }
        anchors = {}
    else:
        curves = {"nasal-profile": _curve(8.0 if camera.view == "left" else 24.0)}
        anchors = {
            name: point
            for name, point in zip(
                ANCHOR_NAMES,
                curves["nasal-profile"][[0, 2, 4, 7]],
            )
        }
    variants = {
        "base": curves,
        "eroded": {
            name: value + np.array([1.0, 0.0])
            for name, value in curves.items()
        },
        "dilated": {
            name: value - np.array([1.0, 0.0])
            for name, value in curves.items()
        },
    }
    variant_rasters = {
        name: _raster(candidate)
        for name, candidate in variants.items()
    }
    named_fields = {
        name: _distance(_raster({name: curve}))
        for name, curve in curves.items()
    }
    boundary = variant_rasters["base"]
    distance = np.minimum.reduce(list(named_fields.values()))
    confidence = np.exp(-distance / 5.0).astype(np.float32)
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=(32, 24),
        mask_canvas_shape=(24, 32),
        work_size=(32, 24),
        roi_work_xyxy=(4.0, 4.0, 28.0, 20.0),
        boundaries_work=curves,
        boundary=boundary,
        distance_fields=named_fields,
        distance_field=distance,
        confidence=confidence,
        variant_boundaries_work=variants,
        variant_boundaries=variant_rasters,
        anchors_work=anchors,
        camera_metadata={
            "camera_name": camera.name,
            "subject_relative_view": semantic_view,
        },
        coordinate_metadata={
            "source_pixel_frame": "undistorted_original_px",
            "observation_pixel_frame": "undistorted_work_px",
            "undistortion_applied": False,
        },
    )


def _bundle() -> NasalObservationBundle:
    return NasalObservationBundle(
        front=_observation("front", _camera("camera2", "front")),
        subject_left=_observation(
            "subject-left",
            _camera("camera1", "left"),
        ),
        subject_right=_observation(
            "subject-right",
            _camera("camera3", "right"),
        ),
    )


def _coordinates(bundle: NasalObservationBundle):
    return {
        observation.camera.view: ObservationCoordinates.from_camera(
            observation.camera,
            work_size=(32, 24),
            pixel_frame="undistorted",
        )
        for observation in bundle.by_view.values()
    }


def _baseline_priors():
    result = {}
    for side, alar_x in (("subject-left", 20.0), ("subject-right", 12.0)):
        result[side] = {}
        for view in ("left", "front", "right"):
            result[side][view] = {
                "upper_tip": [16.0, 7.0],
                "tip_apex": [16.0, 10.0],
                "lower_tip": [16.0, 13.0],
                "alar_transition": [alar_x, 16.0],
            }
    return result


def test_bundle_json_npz_serialization_preserves_subject_semantics(tmp_path):
    json_path = tmp_path / "nasal_observations.json"
    fields_path = tmp_path / "nasal_observation_fields.npz"

    payload = write_nasal_observation_data(
        _bundle(),
        json_path,
        fields_path,
        metadata={
            "rig": {
                "calibration_path": "rig.json",
                "sha256": "abc123",
            }
        },
    )

    restored = json.loads(json_path.read_text(encoding="utf-8"))
    assert payload == restored
    assert restored["camera_name_by_view"] == {
        "front": "camera2",
        "subject-left": "camera1",
        "subject-right": "camera3",
    }
    assert set(restored["views"]) == {
        "front",
        "subject-left",
        "subject-right",
    }
    assert set(restored["views"]["front"]["boundaries_work"]) == {
        "subject-left-alar",
        "subject-right-alar",
    }
    assert set(restored["views"]["subject-left"]["anchors_work"]) == set(
        ANCHOR_NAMES
    )
    assert restored["views"]["subject-left"]["camera"]["camera_name"] == "camera1"
    assert restored["views"]["subject-right"]["camera"]["camera_name"] == "camera3"
    assert restored["fields_npz"] == "nasal_observation_fields.npz"
    assert "confidence" not in restored["views"]["front"]

    with np.load(fields_path, allow_pickle=False) as fields:
        left_key = restored["views"]["subject-left"]["fields"]["confidence"]
        right_key = restored["views"]["subject-right"]["fields"]["confidence"]
        assert fields[left_key].shape == (24, 32)
        assert fields[right_key].shape == (24, 32)
        assert left_key != right_key


def test_static_html_and_all_report_images_are_generated_and_referenced(
    tmp_path,
):
    bundle = _bundle()
    image = np.full((24, 32, 3), 80, dtype=np.uint8)
    masks = {
        "left": np.pad(
            np.ones((16, 20), dtype=np.uint8) * 255,
            ((4, 4), (6, 6)),
        ),
        "right": np.pad(
            np.ones((16, 20), dtype=np.uint8) * 255,
            ((4, 4), (6, 6)),
        ),
    }

    index_path = write_nasal_observation_report(
        tmp_path,
        bundle,
        images_by_view={
            "left": image.copy(),
            "front": image.copy(),
            "right": image.copy(),
        },
        side_face_masks=masks,
        baseline_priors_by_subject_side=_baseline_priors(),
        coordinates_by_view=_coordinates(bundle),
        centerline_x_original=16.0,
    )

    document = index_path.read_text(encoding="utf-8")
    references = re.findall(r"<img[^>]+src=\"([^\"]+)\"", document)
    assert references
    assert "纯几何观测，不含纹理评分" in document
    assert "https://" not in document
    assert "http://" not in document
    assert "epipolar" in document.lower()
    assert "ROI" in document
    assert all((tmp_path / reference).is_file() for reference in references)
    assert {
        "front_overlay.png",
        "left_overlay.png",
        "right_overlay.png",
        "front_confidence.png",
        "left_confidence.png",
        "right_confidence.png",
        "front_variants.png",
        "left_variants.png",
        "right_variants.png",
    }.issubset(set(references))

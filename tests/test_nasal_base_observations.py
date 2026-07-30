from __future__ import annotations

import numpy as np

from src.geometry.nasal_base_observations import (
    NASAL_BASE_ANCHOR_NAMES,
    build_nasal_base_observations,
)


def _landmarks() -> np.ndarray:
    points = np.full((478, 2), 32.0, dtype=np.float64)
    points[75] = (20.0, 35.0)
    points[97] = (27.0, 36.0)
    points[2] = (32.0, 38.0)
    points[326] = (37.0, 36.0)
    points[305] = (44.0, 35.0)
    return points


def _image() -> np.ndarray:
    image = np.full((64, 64, 3), 180, dtype=np.uint8)
    image[35:39, 18:30] = 30
    image[35:39, 35:47] = 30
    return image


def test_front_observation_has_five_subject_semantic_anchors():
    images = {
        "front": _image(),
        "subject-left": _image(),
        "subject-right": _image(),
    }
    landmarks = {name: _landmarks() for name in images}

    bundle = build_nasal_base_observations(
        images,
        landmarks,
        refine_to_image=False,
    )

    assert bundle.front.anchor_names == NASAL_BASE_ANCHOR_NAMES
    assert tuple(bundle.front.landmark_68_indices) == (31, 32, 33, 34, 35)
    np.testing.assert_array_equal(
        bundle.front.target_xy,
        landmarks["front"][[75, 97, 2, 326, 305]],
    )
    assert bundle.subject_left.anchor_names == (
        "columella",
        "subject_left_inner",
        "subject_left_outer",
    )
    assert bundle.subject_right.anchor_names == (
        "subject_right_outer",
        "subject_right_inner",
        "columella",
    )


def test_image_refinement_is_local_finite_and_confidence_weighted():
    images = {
        "front": _image(),
        "subject-left": _image(),
        "subject-right": _image(),
    }
    landmarks = {name: _landmarks() for name in images}

    bundle = build_nasal_base_observations(
        images,
        landmarks,
        refine_to_image=True,
    )

    for observation in bundle.by_view.values():
        assert np.isfinite(observation.target_xy).all()
        assert np.isfinite(observation.confidence).all()
        assert np.all((observation.confidence > 0.0) & (observation.confidence <= 1.0))
        source = landmarks[observation.semantic_view][
            observation.mediapipe_indices
        ]
        assert np.max(np.linalg.norm(observation.target_xy - source, axis=1)) <= 6.0


def test_observation_snapshots_inputs():
    images = {
        "front": _image(),
        "subject-left": _image(),
        "subject-right": _image(),
    }
    landmarks = {name: _landmarks() for name in images}
    bundle = build_nasal_base_observations(
        images,
        landmarks,
        refine_to_image=False,
    )
    landmarks["front"][75] = (0.0, 0.0)

    assert not np.array_equal(bundle.front.target_xy[0], (0.0, 0.0))

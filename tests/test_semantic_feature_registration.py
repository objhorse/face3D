from __future__ import annotations

import numpy as np

from src.appearance.semantic_feature_registration import (
    build_nose_controls,
    detect_nostril_observations,
)


def _landmarks() -> tuple[np.ndarray, np.ndarray]:
    model = np.zeros((68, 2), dtype=np.float32)
    observed = np.zeros((68, 2), dtype=np.float32)
    model[27:36] = np.array(
        [
            [50, 28], [50, 36], [50, 44], [50, 52],
            [34, 62], [42, 66], [50, 68], [58, 66], [66, 62],
        ],
        dtype=np.float32,
    )
    observed[27:36] = np.array(
        [
            [54, 30], [54, 38], [54, 46], [54, 54],
            [34, 64], [44, 69], [54, 71], [64, 69], [74, 64],
        ],
        dtype=np.float32,
    )
    return model, observed


def _nose_mask() -> np.ndarray:
    yy, xx = np.mgrid[:100, :110]
    mask = (((xx - 54.0) / 23.0) ** 2 + ((yy - 54.0) / 20.0) ** 2 <= 1.0)
    return mask.astype(np.uint8) * 255


def test_front_nose_controls_keep_left_and_right_curves_ordered() -> None:
    model, observed = _landmarks()
    controls = build_nose_controls(
        model,
        observed,
        _nose_mask(),
        view="front",
        samples_per_side=6,
    )

    left = controls.group("lower_left")
    right = controls.group("lower_right")
    assert len(left.model_points) == 6
    assert len(right.model_points) == 6
    assert np.all(np.diff(left.model_points[:, 0]) >= 0)
    assert np.all(np.diff(left.observed_points[:, 0]) >= 0)
    assert np.all(np.diff(right.model_points[:, 0]) >= 0)
    assert np.all(np.diff(right.observed_points[:, 0]) >= 0)
    assert np.all(left.observed_points[:, 0] <= observed[33, 0] + 2.0)
    assert np.all(right.observed_points[:, 0] >= observed[33, 0] - 2.0)


def test_missing_mask_uses_only_named_landmarks_with_low_confidence() -> None:
    model, observed = _landmarks()
    controls = build_nose_controls(
        model,
        observed,
        np.zeros((100, 110), dtype=np.uint8),
        view="front",
    )

    assert controls.diagnostics["boundary_control_count"] == 0
    assert controls.confidence < 0.5
    assert set(controls.groups) == {"tip_anchor"}


def test_side_view_never_uses_unordered_boundary_matching() -> None:
    model, observed = _landmarks()
    controls = build_nose_controls(
        model,
        observed,
        _nose_mask(),
        view="left",
    )

    assert controls.groups == ("named_landmarks",) * 9
    np.testing.assert_array_equal(controls.model_points, model[27:36])
    np.testing.assert_array_equal(controls.observed_points, observed[27:36])


def test_detects_two_ordered_nostrils_inside_nose_mask() -> None:
    _model, observed = _landmarks()
    image = np.full((100, 110, 3), 180, dtype=np.uint8)
    yy, xx = np.mgrid[:100, :110]
    left = ((xx - 43.0) / 5.0) ** 2 + ((yy - 59.0) / 3.0) ** 2 <= 1.0
    right = ((xx - 65.0) / 5.0) ** 2 + ((yy - 59.0) / 3.0) ** 2 <= 1.0
    image[left | right] = 25

    result = detect_nostril_observations(image, observed, _nose_mask())

    assert result.confidence >= 0.45
    assert result.centers.shape == (2, 2)
    assert result.centers[0, 0] < observed[33, 0] < result.centers[1, 0]
    np.testing.assert_allclose(result.centers[:, 1], [59, 59], atol=3.0)


def test_uniform_nose_does_not_fabricate_nostril_controls() -> None:
    model, observed = _landmarks()
    image = np.full((100, 110, 3), 180, dtype=np.uint8)
    result = detect_nostril_observations(image, observed, _nose_mask())
    controls = build_nose_controls(
        model,
        observed,
        _nose_mask(),
        view="front",
        image=image,
    )

    assert result.confidence == 0.0
    assert "nostril_left" not in controls.groups
    assert "nostril_right" not in controls.groups

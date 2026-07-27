from __future__ import annotations

import cv2
import inspect
import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalObservationConfig,
    _ordered_epipolar_candidate,
    build_front_nasal_observation as _build_front_nasal_observation,
    build_multiview_nasal_observations as _build_multiview_nasal_observations,
    build_profile_nasal_observation as _build_profile_nasal_observation,
    canvas_points_to_original,
    nasal_view_for_camera,
    original_points_to_canvas,
    original_points_to_work,
    work_points_to_original,
)
from src.geometry.observation_coordinates import ObservationCoordinates
from src.geometry.profile_triangulation import ProfileRig


def _camera(
    name: str,
    view: str,
    image_size: tuple[int, int] = (160, 120),
    center: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> Camera:
    width, height = image_size
    rotation = np.eye(3, dtype=np.float64)
    camera_center = np.asarray(center, dtype=np.float64)
    return Camera(
        name=name,
        view=view,
        image_size=image_size,
        K=np.array(
            [
                [140.0, 0.0, width / 2.0],
                [0.0, 140.0, height / 2.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=rotation,
        t_rig_to_camera=-rotation @ camera_center,
    )


def _rig(*, side_height: float = 0.0) -> ProfileRig:
    return ProfileRig(
        cameras_by_view={
            "left": _camera(
                "camera1",
                "left",
                center=(-0.10, side_height, 0.0),
            ),
            "front": _camera("camera2", "front"),
            "right": _camera(
                "camera3",
                "right",
                center=(0.10, side_height, 0.0),
            ),
        },
        reference_view="front",
        units="meters",
        calibration_path="synthetic",
        stereo_rms_px={"left": 0.1, "front": 0.0, "right": 0.1},
    )


def _config() -> NasalObservationConfig:
    return NasalObservationConfig(
        work_size=(160, 120),
        mask_perturbation_px=2,
        distance_clip_px=24.0,
        min_boundary_points=8,
        max_side_prior_distance_px=12.0,
        max_epipolar_distance_px=4.0,
    )


def _coordinates(
    camera: Camera,
    config: NasalObservationConfig,
    *,
    pixel_frame: str = "undistorted",
) -> ObservationCoordinates:
    return ObservationCoordinates.from_camera(
        camera,
        work_size=config.work_size,
        pixel_frame=pixel_frame,
    )


def _rig_coordinates(
    rig: ProfileRig,
    config: NasalObservationConfig,
) -> dict[str, ObservationCoordinates]:
    return {
        view: _coordinates(camera, config)
        for view, camera in rig.cameras_by_view.items()
    }


def build_front_nasal_observation(
    image,
    mask,
    camera,
    *,
    coordinates=None,
    config=None,
    **kwargs,
):
    limits = config or NasalObservationConfig()
    contract = coordinates or _coordinates(camera, limits)
    return _build_front_nasal_observation(
        image,
        mask,
        camera,
        contract,
        config=limits,
        **kwargs,
    )


def build_profile_nasal_observation(
    image,
    mask,
    rig,
    front_anchors,
    side_view,
    side_prior_original,
    *,
    coordinates_by_view=None,
    config=None,
    **kwargs,
):
    limits = config or NasalObservationConfig()
    contracts = coordinates_by_view or _rig_coordinates(rig, limits)
    return _build_profile_nasal_observation(
        image,
        mask,
        rig,
        front_anchors,
        side_view,
        side_prior_original,
        coordinates_by_view=contracts,
        config=limits,
        **kwargs,
    )


def build_multiview_nasal_observations(*, coordinates_by_view=None, **kwargs):
    limits = kwargs.get("config") or NasalObservationConfig()
    rig = kwargs["rig"]
    contracts = coordinates_by_view or _rig_coordinates(rig, limits)
    return _build_multiview_nasal_observations(
        coordinates_by_view=contracts,
        **kwargs,
    )


def _front_mask(*, jagged: bool = False) -> np.ndarray:
    mask = np.zeros((120, 160), dtype=np.uint8)
    contour = np.array(
        [
            [72, 24],
            [58, 38],
            [51, 56],
            [37, 72],
            [47, 88],
            [65, 95],
            [82, 97],
            [104, 91],
            [112, 76],
            [105, 58],
            [92, 37],
        ],
        dtype=np.int32,
    )
    cv2.fillPoly(mask, [contour], 255)
    if jagged:
        for y in range(56, 91, 6):
            cv2.rectangle(mask, (35, y), (45, y + 2), 255, -1)
            cv2.rectangle(mask, (106, y + 3), (119, y + 5), 255, -1)
    return mask


def _vertical_front_mask() -> np.ndarray:
    mask = np.zeros((120, 160), dtype=np.uint8)
    cv2.rectangle(mask, (46, 18), (114, 102), 255, -1)
    return mask


def _profile_mask(side: str, *, jagged: bool = False) -> np.ndarray:
    mask = np.zeros((120, 160), dtype=np.uint8)
    left_outline = np.array(
        [
            [68, 12],
            [66, 28],
            [60, 39],
            [52, 47],
            [43, 57],
            [48, 67],
            [55, 76],
            [61, 86],
            [66, 94],
            [60, 105],
            [75, 114],
            [151, 116],
            [151, 8],
        ],
        dtype=np.int32,
    )
    outline = (
        left_outline
        if side == "subject-left"
        else np.column_stack((159 - left_outline[:, 0], left_outline[:, 1]))
    )
    cv2.fillPoly(mask, [outline], 255)
    if jagged:
        direction = -1 if side == "subject-left" else 1
        base_x = 52 if side == "subject-left" else 107
        for index, y in enumerate(range(39, 92, 5)):
            length = 5 + 3 * (index % 3)
            x1 = base_x + direction * length
            cv2.line(mask, (base_x, y), (x1, y), 255, 2)
    return mask


def _letterbox_mask(mask: np.ndarray, size: int) -> np.ndarray:
    image_height, image_width = mask.shape[:2]
    resized_height = int(image_height * size / image_width)
    resized = cv2.resize(
        mask,
        (size, resized_height),
        interpolation=cv2.INTER_NEAREST,
    )
    canvas = np.zeros((size, size), dtype=np.uint8)
    y0 = (size - resized_height) // 2
    canvas[y0 : y0 + resized_height] = resized
    return canvas


def _profile_priors(side: str) -> dict[str, tuple[float, float]]:
    left = {
        "upper_tip": (59.0, 40.0),
        "tip_apex": (44.0, 57.0),
        "lower_tip": (53.0, 74.0),
        "alar_transition": (63.0, 88.0),
    }
    if side == "subject-left":
        return left
    return {name: (159.0 - x, y) for name, (x, y) in left.items()}


def _front_anchors() -> dict[str, tuple[float, float]]:
    return {
        "upper_tip": (75.0, 40.0),
        "tip_apex": (76.0, 56.0),
        "lower_tip": (76.0, 74.0),
        "alar_transition": (77.0, 88.0),
    }


def _boundary_confidence(observation) -> float:
    values = observation.confidence[observation.boundary]
    assert len(values)
    return float(np.mean(values))


def test_coordinate_spaces_round_trip_without_reimplementing_letterbox_logic():
    points = np.array([[0.0, 0.0], [80.0, 60.0], [159.0, 119.0]])
    canvas = original_points_to_canvas(points, (160, 120), (128, 128))
    restored = canvas_points_to_original(canvas, (160, 120), (128, 128))
    coordinates = _coordinates(
        _camera("camera2", "front"),
        NasalObservationConfig(work_size=(80, 60)),
    )
    work = original_points_to_work(points, coordinates)
    restored_from_work = work_points_to_original(work, coordinates)

    assert restored == pytest.approx(points, abs=1e-8)
    assert restored_from_work == pytest.approx(points, abs=1e-8)
    assert canvas[0, 1] == pytest.approx(16.0)


@pytest.mark.parametrize(
    ("name", "camera_view", "nasal_view"),
    [
        ("camera1", "left", "subject-left"),
        ("camera2", "front", "front"),
        ("camera3", "right", "subject-right"),
    ],
)
def test_camera_names_map_to_subject_semantics(name, camera_view, nasal_view):
    assert nasal_view_for_camera(_camera(name, camera_view)) == nasal_view


def test_camera_mapping_rejects_mislabeled_fixed_rig_camera():
    with pytest.raises(ValueError, match="camera1.*left"):
        nasal_view_for_camera(_camera("camera1", "right"))


def test_front_observation_preserves_asymmetric_subject_alar_boundaries():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    observation = build_front_nasal_observation(
        image,
        _front_mask(),
        _camera("camera2", "front"),
        centerline_x_original=76.0,
        config=_config(),
    )

    subject_left = observation.boundaries_work["subject-left-alar"]
    subject_right = observation.boundaries_work["subject-right-alar"]

    assert observation.semantic_view == "front"
    assert np.median(subject_left[:, 0]) > 76.0
    assert np.median(subject_right[:, 0]) < 76.0
    assert np.ptp(subject_left[:, 0]) > 0.0
    assert 76.0 - np.min(subject_right[:, 0]) > np.max(subject_left[:, 0]) - 76.0
    assert observation.distance_field.shape == (120, 160)
    assert np.all(observation.distance_field[observation.boundary] == 0.0)


def test_front_distance_fields_keep_subject_sides_separate():
    observation = build_front_nasal_observation(
        np.zeros((120, 160, 3), dtype=np.uint8),
        _front_mask(),
        _camera("camera2", "front"),
        centerline_x_original=76.0,
        config=_config(),
    )
    fields = observation.distance_fields
    left_point = np.rint(
        observation.boundaries_work["subject-left-alar"][10]
    ).astype(int)
    right_point = np.rint(
        observation.boundaries_work["subject-right-alar"][10]
    ).astype(int)

    assert set(fields) == {"subject-left-alar", "subject-right-alar"}
    assert fields["subject-left-alar"][left_point[1], left_point[0]] == 0.0
    assert fields["subject-right-alar"][left_point[1], left_point[0]] > 10.0
    assert fields["subject-right-alar"][right_point[1], right_point[0]] == 0.0
    assert fields["subject-left-alar"][right_point[1], right_point[0]] > 10.0
    assert observation.coordinate_metadata[
        "aggregate_distance_field_usage"
    ] == "display_only"
    assert observation.distance_field == pytest.approx(
        np.minimum(
            fields["subject-left-alar"],
            fields["subject-right-alar"],
        )
    )


def test_front_confidence_drops_for_unstable_mask_perturbations():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    camera = _camera("camera2", "front")
    stable = build_front_nasal_observation(
        image,
        _front_mask(),
        camera,
        centerline_x_original=76.0,
        config=_config(),
    )
    unstable = build_front_nasal_observation(
        image,
        _front_mask(jagged=True),
        camera,
        centerline_x_original=76.0,
        config=_config(),
    )

    assert _boundary_confidence(stable) > _boundary_confidence(unstable) + 0.03
    unstable_values = unstable.confidence[unstable.boundary]
    assert np.any((unstable_values > 0.0) & (unstable_values < 1.0))


def test_local_image_gradient_only_boosts_boundary_confidence():
    camera = _camera("camera2", "front")
    mask = _front_mask()
    flat = np.zeros((120, 160, 3), dtype=np.uint8)
    edged = flat.copy()
    contour, _hierarchy = cv2.findContours(
        mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
    )
    cv2.drawContours(edged, contour, -1, (255, 255, 255), 2)

    flat_observation = build_front_nasal_observation(
        flat,
        mask,
        camera,
        centerline_x_original=76.0,
        config=_config(),
    )
    edged_observation = build_front_nasal_observation(
        edged,
        mask,
        camera,
        centerline_x_original=76.0,
        config=_config(),
    )

    assert np.array_equal(flat_observation.boundary, edged_observation.boundary)
    assert np.array_equal(
        flat_observation.distance_field,
        edged_observation.distance_field,
    )
    assert _boundary_confidence(edged_observation) > _boundary_confidence(
        flat_observation
    )


def test_strong_tangential_gradient_does_not_boost_confidence():
    camera = _camera("camera2", "front")
    mask = _vertical_front_mask()
    flat = np.zeros((120, 160, 3), dtype=np.uint8)
    normal = np.repeat(mask[..., None], 3, axis=2)
    vertical_ramp = np.linspace(0, 255, 120, dtype=np.uint8)[:, None]
    tangential = np.repeat(vertical_ramp, 160, axis=1)
    tangential = np.repeat(tangential[..., None], 3, axis=2)

    observations = [
        build_front_nasal_observation(
            image,
            mask,
            camera,
            centerline_x_original=80.0,
            config=_config(),
        )
        for image in (flat, normal, tangential)
    ]
    flat_confidence, normal_confidence, tangential_confidence = (
        _boundary_confidence(observation)
        for observation in observations
    )

    assert normal_confidence > flat_confidence + 0.05
    assert tangential_confidence == pytest.approx(flat_confidence, abs=0.01)


def test_rgb_is_default_and_bgr_requires_explicit_configuration():
    camera = _camera("camera2", "front")
    mask = _vertical_front_mask()
    flat = np.zeros((120, 160, 3), dtype=np.uint8)
    red_edge = flat.copy()
    red_edge[..., 0] = mask
    rgba_edge = np.dstack(
        (red_edge, np.full(mask.shape, 255, dtype=np.uint8))
    )
    rgb_config = NasalObservationConfig(
        **{
            **_config().__dict__,
            "gradient_noise_floor": 150.0,
        }
    )
    bgr_config = NasalObservationConfig(
        **{
            **rgb_config.__dict__,
            "color_space": "BGR",
        }
    )

    flat_observation = build_front_nasal_observation(
        flat,
        mask,
        camera,
        centerline_x_original=80.0,
        config=rgb_config,
    )
    rgb_observation = build_front_nasal_observation(
        red_edge,
        mask,
        camera,
        centerline_x_original=80.0,
        config=rgb_config,
    )
    bgr_observation = build_front_nasal_observation(
        red_edge,
        mask,
        camera,
        centerline_x_original=80.0,
        config=bgr_config,
    )
    rgba_observation = build_front_nasal_observation(
        rgba_edge,
        mask,
        camera,
        centerline_x_original=80.0,
        config=rgb_config,
    )

    assert rgb_observation.coordinate_metadata["color_space"] == "RGB"
    assert _boundary_confidence(rgba_observation) == pytest.approx(
        _boundary_confidence(rgb_observation),
        abs=0.01,
    )
    assert _boundary_confidence(rgb_observation) > _boundary_confidence(
        flat_observation
    ) + 0.03
    assert _boundary_confidence(bgr_observation) == pytest.approx(
        _boundary_confidence(flat_observation),
        abs=0.01,
    )


def test_low_contrast_noise_does_not_raise_gradient_confidence():
    camera = _camera("camera2", "front")
    mask = _front_mask()
    flat = np.full((120, 160, 3), 127, dtype=np.uint8)
    rng = np.random.default_rng(7)
    noise = rng.integers(-1, 2, size=flat.shape, dtype=np.int16)
    noisy = np.clip(flat.astype(np.int16) + noise, 0, 255).astype(np.uint8)

    observations = [
        build_front_nasal_observation(
            image,
            mask,
            camera,
            centerline_x_original=76.0,
            config=_config(),
        )
        for image in (flat, noisy)
    ]

    assert _boundary_confidence(observations[1]) == pytest.approx(
        _boundary_confidence(observations[0]),
        abs=0.01,
    )


def test_strong_edge_outside_nasal_roi_does_not_change_normalization():
    camera = _camera("camera2", "front")
    mask = _front_mask()
    weak = np.zeros((120, 160, 3), dtype=np.uint8)
    contour, _hierarchy = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    cv2.drawContours(weak, contour, -1, (24, 24, 24), 2)
    outside = weak.copy()
    outside[:, :8] = 255

    observations = [
        build_front_nasal_observation(
            image,
            mask,
            camera,
            centerline_x_original=76.0,
            config=_config(),
        )
        for image in (weak, outside)
    ]

    assert _boundary_confidence(observations[1]) == pytest.approx(
        _boundary_confidence(observations[0]),
        abs=0.005,
    )


def test_ordered_anchor_dp_finds_legal_solution_rejected_by_greedy_choice():
    curve = np.column_stack(
        (np.arange(8, dtype=np.float64), np.zeros(8, dtype=np.float64))
    )
    target_x = {
        "upper_tip": 2.0,
        "tip_apex": 1.0,
        "lower_tip": 4.0,
        "alar_transition": 5.0,
    }
    lines = {
        name: np.array([1.0, 0.0, -x], dtype=np.float64)
        for name, x in target_x.items()
    }
    priors = {
        name: np.array([x, 0.0], dtype=np.float64)
        for name, x in target_x.items()
    }
    config = NasalObservationConfig(
        work_size=(8, 2),
        min_boundary_points=4,
        max_epipolar_distance_px=2.1,
        max_side_prior_distance_px=2.1,
    )
    greedy_indices = [
        int(np.argmin(np.abs(curve[:, 0] - target_x[name])))
        for name in ("upper_tip", "tip_apex", "lower_tip", "alar_transition")
    ]

    result = _ordered_epipolar_candidate(curve, lines, priors, config)

    assert greedy_indices == [2, 1, 4, 5]
    assert result is not None
    matched_curve, anchors, _epipolar_errors, _prior_errors = result
    indices = [
        int(np.argmin(np.linalg.norm(matched_curve - anchors[name], axis=1)))
        for name in ("upper_tip", "tip_apex", "lower_tip", "alar_transition")
    ]
    assert np.all(np.diff(indices) > 0)


def test_ordered_anchor_dp_preserves_long_span_path_to_same_endpoint():
    curve = np.column_stack(
        (np.arange(12, dtype=np.float64), np.zeros(12, dtype=np.float64))
    )
    target_x = {
        "upper_tip": 8.0,
        "tip_apex": 9.0,
        "lower_tip": 10.0,
        "alar_transition": 11.0,
    }
    lines = {
        name: np.array([1.0, 0.0, -x], dtype=np.float64)
        for name, x in target_x.items()
    }
    priors = {
        name: np.array([x, 0.0], dtype=np.float64)
        for name, x in target_x.items()
    }
    config = NasalObservationConfig(
        work_size=(12, 2),
        min_boundary_points=12,
        max_epipolar_distance_px=8.1,
        max_side_prior_distance_px=8.1,
    )

    result = _ordered_epipolar_candidate(curve, lines, priors, config)

    assert result is not None
    matched_curve, anchors, _epipolar_errors, _prior_errors = result
    assert len(matched_curve) == 12
    assert anchors["upper_tip"] == pytest.approx([0.0, 0.0])
    assert anchors["tip_apex"] == pytest.approx([9.0, 0.0])
    assert anchors["lower_tip"] == pytest.approx([10.0, 0.0])
    assert anchors["alar_transition"] == pytest.approx([11.0, 0.0])


def test_front_confidence_is_stable_across_letterbox_mask_resolutions():
    camera = _camera("camera2", "front")
    config = _config()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    observations = [
        build_front_nasal_observation(
            image,
            _letterbox_mask(_front_mask(), size),
            camera,
            centerline_x_original=76.0,
            config=config,
        )
        for size in (160, 640, 1024)
    ]

    confidence = np.asarray(
        [_boundary_confidence(observation) for observation in observations]
    )
    reference_curve = observations[0].boundaries_work["subject-left-alar"]
    for observation in observations[1:]:
        curve = observation.boundaries_work["subject-left-alar"]
        distances = np.linalg.norm(
            reference_curve[:, None, :] - curve[None, :, :],
            axis=2,
        )
        assert float(np.percentile(np.min(distances, axis=1), 95.0)) <= 1.5
    assert float(np.ptp(confidence)) <= 0.02


def test_observation_arrays_and_mappings_are_deeply_immutable():
    observation = build_front_nasal_observation(
        np.zeros((120, 160, 3), dtype=np.uint8),
        _front_mask(),
        _camera("camera2", "front"),
        centerline_x_original=76.0,
        config=_config(),
    )
    curve = observation.boundaries_work["subject-left-alar"]
    base_variant = observation.variant_boundaries_work["base"][
        "subject-left-alar"
    ]

    assert not np.shares_memory(curve, base_variant)
    assert not np.shares_memory(
        observation.boundary,
        observation.variant_boundaries["base"],
    )
    with pytest.raises(TypeError):
        observation.boundaries_work["new"] = np.zeros((2, 2))
    with pytest.raises(ValueError):
        curve[0, 0] = 0.0
    with pytest.raises(ValueError):
        observation.distance_fields["subject-left-alar"][0, 0] = 0.0
    with pytest.raises(ValueError):
        observation.camera.K[0, 0] = 0.0
    with pytest.raises(TypeError):
        observation.coordinate_metadata["new"] = "value"
    assert "precise_euclidean" in observation.coordinate_metadata[
        "distance_field"
    ]


@pytest.mark.parametrize(
    ("camera_name", "camera_view", "semantic_view"),
    [
        ("camera1", "left", "subject-left"),
        ("camera3", "right", "subject-right"),
    ],
)
def test_profile_observation_recovers_continuous_local_nasal_curve(
    camera_name,
    camera_view,
    semantic_view,
):
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    observation = build_profile_nasal_observation(
        image,
        _profile_mask(semantic_view),
        _rig(),
        _front_anchors(),
        camera_view,
        side_prior_original=_profile_priors(semantic_view),
        roi_original_xyxy=(28.0, 28.0, 132.0, 100.0),
        config=_config(),
    )
    curve = observation.boundaries_work["nasal-profile"]
    anchors = observation.anchors_work

    assert observation.semantic_view == semantic_view
    assert len(curve) >= 25
    assert set(anchors) == {
        "upper_tip",
        "tip_apex",
        "lower_tip",
        "alar_transition",
    }
    assert anchors["upper_tip"][1] < anchors["tip_apex"][1]
    assert anchors["tip_apex"][1] < anchors["alar_transition"][1]
    assert np.max(curve[:, 1]) < 101.0
    assert np.min(curve[:, 1]) > 27.0
    steps = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    assert np.percentile(steps, 95.0) <= 2.0


def test_profile_variants_and_point_order_survive_small_mask_perturbation():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    rig = _rig()
    base = build_profile_nasal_observation(
        image,
        _profile_mask("subject-left"),
        rig,
        _front_anchors(),
        "left",
        _profile_priors("subject-left"),
        config=_config(),
    )
    shifted_mask = cv2.warpAffine(
        _profile_mask("subject-left"),
        np.array([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]]),
        (160, 120),
        flags=cv2.INTER_NEAREST,
    )
    shifted = build_profile_nasal_observation(
        image,
        shifted_mask,
        rig,
        _front_anchors(),
        "left",
        _profile_priors("subject-left"),
        config=_config(),
    )

    assert set(base.variant_boundaries_work) == {"eroded", "base", "dilated"}
    assert set(base.variant_boundaries) == {"eroded", "base", "dilated"}
    assert np.array_equal(base.variant_boundaries["base"], base.boundary)
    curve = base.boundaries_work["nasal-profile"]
    anchor_indices = [
        int(np.argmin(np.linalg.norm(curve - base.anchors_work[name], axis=1)))
        for name in (
            "upper_tip",
            "tip_apex",
            "lower_tip",
            "alar_transition",
        )
    ]
    assert anchor_indices == sorted(anchor_indices)
    shifted_curve = shifted.boundaries_work["nasal-profile"]
    pairwise = np.linalg.norm(
        curve[:, None, :] - shifted_curve[None, :, :],
        axis=2,
    )
    assert float(np.percentile(np.min(pairwise, axis=1), 90.0)) <= 2.0


def test_epipolar_evidence_selects_nose_over_mouth_and_rear_cheek():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    rig = _rig()

    left = build_profile_nasal_observation(
        image,
        _profile_mask("subject-left"),
        rig,
        _front_anchors(),
        "left",
        _profile_priors("subject-left"),
        config=_config(),
    )
    right = build_profile_nasal_observation(
        image,
        _profile_mask("subject-right"),
        rig,
        _front_anchors(),
        "right",
        _profile_priors("subject-right"),
        config=_config(),
    )

    left_curve = left.boundaries_work["nasal-profile"]
    right_curve = right.boundaries_work["nasal-profile"]
    assert np.median(left_curve[:, 0]) < 90.0
    assert np.median(right_curve[:, 0]) > 70.0
    assert np.max(left_curve[:, 1]) < 96.0
    assert np.max(right_curve[:, 1]) < 96.0
    assert left.coordinate_metadata["candidate_count"] == 1
    assert right.coordinate_metadata["candidate_count"] == 1
    assert max(
        left.coordinate_metadata["selected_anchor_prior_errors_px"]
    ) <= _config().max_side_prior_distance_px


def test_rig_epipolar_change_can_invalidate_joint_prior_support():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = _profile_mask("subject-left")
    baseline = build_profile_nasal_observation(
        image,
        mask,
        _rig(),
        _front_anchors(),
        "left",
        _profile_priors("subject-left"),
        config=_config(),
    )

    with pytest.raises(ValueError, match="side prior.*epipolar"):
        build_profile_nasal_observation(
            image,
            mask,
            _rig(side_height=0.08),
            _front_anchors(),
            "left",
            _profile_priors("subject-left"),
            config=_config(),
        )

    baseline_lines = np.asarray(
        baseline.coordinate_metadata["epipolar_lines_work"]
    )
    assert np.isfinite(baseline_lines).all()


def test_profile_rejects_front_anchors_with_no_epipolar_intersection():
    anchors = {
        "upper_tip": (75.0, 0.0),
        "tip_apex": (76.0, 1.0),
        "lower_tip": (76.0, 2.0),
        "alar_transition": (77.0, 3.0),
    }
    with pytest.raises(ValueError, match="epipolar"):
        build_profile_nasal_observation(
            np.zeros((120, 160, 3), dtype=np.uint8),
            _profile_mask("subject-left"),
            _rig(),
            anchors,
            "left",
            _profile_priors("subject-left"),
            config=_config(),
        )


def test_production_apis_require_projected_side_priors():
    profile_parameter = inspect.signature(
        _build_profile_nasal_observation
    ).parameters["side_prior_original"]
    multiview_parameter = inspect.signature(
        _build_multiview_nasal_observations
    ).parameters["side_priors_original"]
    front_coordinates = inspect.signature(
        _build_front_nasal_observation
    ).parameters["coordinates"]
    profile_coordinates = inspect.signature(
        _build_profile_nasal_observation
    ).parameters["coordinates_by_view"]
    multiview_coordinates = inspect.signature(
        _build_multiview_nasal_observations
    ).parameters["coordinates_by_view"]

    assert profile_parameter.default is inspect.Parameter.empty
    assert multiview_parameter.default is inspect.Parameter.empty
    assert front_coordinates.default is inspect.Parameter.empty
    assert profile_coordinates.default is inspect.Parameter.empty
    assert multiview_coordinates.default is inspect.Parameter.empty


def test_multiview_api_rejects_missing_projected_side_prior():
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="side projected priors.*right"):
        build_multiview_nasal_observations(
            images_by_view={
                "left": image,
                "front": image,
                "right": image,
            },
            front_semantic_nose_mask=_front_mask(),
            side_face_masks={
                "left": _profile_mask("subject-left"),
                "right": _profile_mask("subject-right"),
            },
            rig=_rig(),
            front_anchors_original=_front_anchors(),
            side_priors_original={
                "left": _profile_priors("subject-left"),
            },
            centerline_x_original=76.0,
            config=_config(),
        )


def test_production_api_builds_bilateral_observations_with_side_priors():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    bundle = build_multiview_nasal_observations(
        images_by_view={
            "left": image,
            "front": image,
            "right": image,
        },
        front_semantic_nose_mask=_front_mask(),
        side_face_masks={
            "left": _profile_mask("subject-left"),
            "right": _profile_mask("subject-right"),
        },
        rig=_rig(),
        front_anchors_original=_front_anchors(),
        side_priors_original={
            "left": _profile_priors("subject-left"),
            "right": _profile_priors("subject-right"),
        },
        centerline_x_original=76.0,
        config=_config(),
    )

    assert bundle.camera_name_by_view == {
        "front": "camera2",
        "subject-left": "camera1",
        "subject-right": "camera3",
    }
    assert bundle.subject_left.boundaries_work["nasal-profile"].shape[0] >= 20
    assert bundle.subject_right.boundaries_work["nasal-profile"].shape[0] >= 20


def test_side_prior_hard_constraint_rejects_epipolar_only_contour():
    displaced_prior = {
        name: (100.0, y)
        for name, (_x, y) in _profile_priors("subject-left").items()
    }

    with pytest.raises(ValueError, match="side prior"):
        build_profile_nasal_observation(
            np.zeros((120, 160, 3), dtype=np.uint8),
            _profile_mask("subject-left"),
            _rig(),
            _front_anchors(),
            "left",
            displaced_prior,
            config=_config(),
        )


def test_low_confidence_on_one_profile_does_not_contaminate_the_other():
    config = _config()
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    front = build_front_nasal_observation(
        image,
        _front_mask(),
        _camera("camera2", "front"),
        centerline_x_original=76.0,
        config=config,
    )
    subject_left = build_profile_nasal_observation(
        image,
        _profile_mask("subject-left"),
        _rig(),
        _front_anchors(),
        "left",
        side_prior_original=_profile_priors("subject-left"),
        roi_original_xyxy=(28.0, 28.0, 132.0, 100.0),
        config=config,
    )
    left_confidence_before = subject_left.confidence.copy()
    subject_right = build_profile_nasal_observation(
        image,
        _profile_mask("subject-right", jagged=True),
        _rig(),
        _front_anchors(),
        "right",
        side_prior_original=_profile_priors("subject-right"),
        roi_original_xyxy=(28.0, 28.0, 132.0, 100.0),
        config=config,
    )

    bundle = NasalObservationBundle(
        front=front,
        subject_left=subject_left,
        subject_right=subject_right,
    )

    assert np.array_equal(bundle.subject_left.confidence, left_confidence_before)
    assert _boundary_confidence(bundle.subject_left) > _boundary_confidence(
        bundle.subject_right
    )


def test_observation_errors_are_explicit():
    config = _config()
    camera = _camera("camera2", "front")
    image = np.zeros((120, 160, 3), dtype=np.uint8)

    with pytest.raises(ValueError, match="nose mask is empty"):
        build_front_nasal_observation(
            image,
            np.zeros((120, 160), dtype=np.uint8),
            camera,
            centerline_x_original=80.0,
            config=config,
        )
    with pytest.raises(ValueError, match="image size.*camera2"):
        build_front_nasal_observation(
            np.zeros((100, 160, 3), dtype=np.uint8),
            _front_mask(),
            camera,
            centerline_x_original=80.0,
            config=config,
        )
    with pytest.raises(ValueError, match="mask size.*square letterbox"):
        build_front_nasal_observation(
            image,
            np.zeros((90, 140), dtype=np.uint8),
            camera,
            centerline_x_original=80.0,
            config=config,
        )
    with pytest.raises(ValueError, match="ROI.*image bounds"):
        build_profile_nasal_observation(
            image,
            _profile_mask("subject-left"),
            _rig(),
            _front_anchors(),
            "left",
            side_prior_original=_profile_priors("subject-left"),
            roi_original_xyxy=(-1.0, 20.0, 90.0, 100.0),
            config=config,
        )

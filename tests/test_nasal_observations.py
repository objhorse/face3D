from __future__ import annotations

import cv2
import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.nasal_observations import (
    NasalObservationBundle,
    NasalObservationConfig,
    build_front_nasal_observation,
    build_multiview_nasal_observations,
    build_profile_nasal_observation,
    canvas_points_to_original,
    nasal_view_for_camera,
    original_points_to_canvas,
    original_points_to_work,
    work_points_to_original,
)
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
        profile_prior_padding_px=10.0,
        max_epipolar_distance_px=4.0,
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
    work = original_points_to_work(points, (160, 120), (80, 60))
    restored_from_work = work_points_to_original(work, (160, 120), (80, 60))

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
        config=_config(),
    )
    right = build_profile_nasal_observation(
        image,
        _profile_mask("subject-right"),
        rig,
        _front_anchors(),
        "right",
        config=_config(),
    )

    left_curve = left.boundaries_work["nasal-profile"]
    right_curve = right.boundaries_work["nasal-profile"]
    assert np.median(left_curve[:, 0]) < 90.0
    assert np.median(right_curve[:, 0]) > 70.0
    assert np.max(left_curve[:, 1]) < 96.0
    assert np.max(right_curve[:, 1]) < 96.0


def test_rig_epipolar_change_changes_profile_selection():
    image = np.zeros((120, 160, 3), dtype=np.uint8)
    mask = _profile_mask("subject-left")
    baseline = build_profile_nasal_observation(
        image,
        mask,
        _rig(),
        _front_anchors(),
        "left",
        config=_config(),
    )

    changed = build_profile_nasal_observation(
        image,
        mask,
        _rig(side_height=0.08),
        _front_anchors(),
        "left",
        config=_config(),
    )

    baseline_lines = np.asarray(
        baseline.coordinate_metadata["epipolar_lines_work"]
    )
    changed_lines = np.asarray(
        changed.coordinate_metadata["epipolar_lines_work"]
    )
    baseline_curve = baseline.boundaries_work["nasal-profile"]
    changed_curve = changed.boundaries_work["nasal-profile"]
    assert np.isfinite(baseline_lines).all()
    assert not np.allclose(changed_lines, baseline_lines)
    assert abs(
        float(np.mean(changed_curve[:, 1]) - np.mean(baseline_curve[:, 1]))
    ) > 8.0


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
            config=_config(),
        )


def test_production_api_builds_bilateral_observations_without_side_priors():
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

from __future__ import annotations

import numpy as np
import pytest
import json
import cv2

from src.cross_view_geometry import Camera
from src.geometry.profile_triangulation import ProfileRig, project_reference_point
from src.geometry.nasal_texture_observations import (
    NasalEpipolarSeed,
    NasalCoordinateProvenance,
    NasalPairMatch,
    RejectedNasalObservation,
    NasalTextureConfidenceMaps,
    NasalTextureObservationBundle,
    NasalTextureObservationConfig,
    TrustedNasalObservation,
    build_nasal_texture_confidence_maps,
    match_model_guided_nasal_pair,
    triangulate_nasal_pair_matches,
)


def _provenance(view: str) -> NasalCoordinateProvenance:
    return NasalCoordinateProvenance(
        semantic_view=view,
        source_size=(80, 64),
        work_size=(40, 32),
        source_to_work=np.asarray(
            [[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        ),
        undistorted=True,
    )


def _pair_match() -> NasalPairMatch:
    return NasalPairMatch(
        side_view="subject-left",
        front_pixel=np.asarray([20.0, 15.0]),
        side_pixel=np.asarray([18.5, 15.25]),
        confidence=0.9,
        semantic_region="soft_triangle",
        source_matcher="model_guided_census",
        provenance_by_view={
            "front": _provenance("front"),
            "subject-left": _provenance("subject-left"),
        },
        diagnostics={"epipolar_error_px": 0.2, "uniqueness_margin": 0.3},
    )


def test_coordinate_provenance_rejects_singular_transform() -> None:
    with pytest.raises(ValueError, match="source_to_work"):
        NasalCoordinateProvenance(
            semantic_view="front",
            source_size=(80, 64),
            work_size=(40, 32),
            source_to_work=np.zeros((3, 3), dtype=np.float64),
            undistorted=True,
        )

    scaled = NasalCoordinateProvenance(
        semantic_view="front",
        source_size=(80, 64),
        work_size=(40, 32),
        source_to_work=np.diag([0.5, 0.5, 1.0]) * 1e-8,
        undistorted=True,
    )
    assert np.allclose(scaled.source_to_work, np.diag([0.5, 0.5, 1.0]))

    with pytest.raises(ValueError, match="undistorted"):
        NasalCoordinateProvenance(
            semantic_view="front",
            source_size=(80, 64),
            work_size=(40, 32),
            source_to_work=np.diag([0.5, 0.5, 1.0]),
            undistorted="yes",  # type: ignore[arg-type]
        )

    with pytest.raises(ValueError, match="condition"):
        NasalCoordinateProvenance(
            semantic_view="front",
            source_size=(80, 64),
            work_size=(40, 32),
            source_to_work=np.diag([1e12, 1e-12, 1.0]),
            undistorted=True,
        )


def test_pair_match_requires_canonical_front_side_pair() -> None:
    with pytest.raises(ValueError, match="side_view"):
        NasalPairMatch(
            side_view="front",
            front_pixel=np.asarray([20.0, 15.0]),
            side_pixel=np.asarray([18.5, 15.25]),
            confidence=0.9,
            semantic_region="alar_dome",
            source_matcher="loftr",
            provenance_by_view={"front": _provenance("front")},
            diagnostics={},
        )


def test_trusted_observation_validates_geometry_and_is_immutable() -> None:
    observation = TrustedNasalObservation(
        pair_match=_pair_match(),
        point_reference_m=np.asarray([0.01, -0.02, 0.42]),
        covariance_proxy=np.eye(3, dtype=np.float64) * 1e-6,
        depths_m={"front": 0.42, "subject-left": 0.44},
        reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
        ray_angle_deg=14.0,
        weight=0.8,
    )
    with pytest.raises(ValueError):
        observation.point_reference_m[0] = 99.0
    with pytest.raises(ValueError):
        observation.pair_match.front_pixel[0] = 99.0

    with pytest.raises(ValueError, match="positive depth"):
        TrustedNasalObservation(
            pair_match=_pair_match(),
            point_reference_m=np.asarray([0.01, -0.02, -0.42]),
            covariance_proxy=np.eye(3, dtype=np.float64),
            depths_m={"front": -0.42, "subject-left": 0.44},
            reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
            ray_angle_deg=14.0,
            weight=0.8,
        )

    with pytest.raises(ValueError, match="reprojection"):
        TrustedNasalObservation(
            pair_match=_pair_match(),
            point_reference_m=np.asarray([0.01, -0.02, 0.42]),
            covariance_proxy=np.eye(3, dtype=np.float64),
            depths_m={"front": 0.42, "subject-left": 0.44},
            reprojection_errors_px={"front": 0.4, "subject-left": 9.0},
            ray_angle_deg=14.0,
            weight=0.8,
        )

    with pytest.raises(ValueError, match="ray_angle"):
        TrustedNasalObservation(
            pair_match=_pair_match(),
            point_reference_m=np.asarray([0.01, -0.02, 0.42]),
            covariance_proxy=np.eye(3, dtype=np.float64),
            depths_m={"front": 0.42, "subject-left": 0.44},
            reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
            ray_angle_deg=1.0,
            weight=0.8,
        )

    with pytest.raises(ValueError, match="reference depth"):
        TrustedNasalObservation(
            pair_match=_pair_match(),
            point_reference_m=np.asarray([0.01, -0.02, 99.0]),
            covariance_proxy=np.eye(3, dtype=np.float64),
            depths_m={"front": 0.42, "subject-left": 0.44},
            reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
            ray_angle_deg=14.0,
            weight=0.8,
        )

    low_confidence = NasalPairMatch(
        **{
            **_pair_match().__dict__,
            "confidence": 0.1,
        }
    )
    with pytest.raises(ValueError, match="match confidence"):
        TrustedNasalObservation(
            pair_match=low_confidence,
            point_reference_m=np.asarray([0.01, -0.02, 0.42]),
            covariance_proxy=np.eye(3, dtype=np.float64),
            depths_m={"front": 0.42, "subject-left": 0.44},
            reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
            ray_angle_deg=14.0,
            weight=0.8,
        )


def test_bundle_requires_bilateral_trusted_observations() -> None:
    left = TrustedNasalObservation(
        pair_match=_pair_match(),
        point_reference_m=np.asarray([0.01, -0.02, 0.42]),
        covariance_proxy=np.eye(3, dtype=np.float64) * 1e-6,
        depths_m={"front": 0.42, "subject-left": 0.44},
        reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
        ray_angle_deg=14.0,
        weight=0.8,
    )
    rejected_diagnostics = {"candidates": [1, 2]}
    rejected = RejectedNasalObservation(
        pair_match=NasalPairMatch(
            **{
                **_pair_match().__dict__,
                "diagnostics": {"epipolar_error_px": 8.0},
            }
        ),
        reasons=("epipolar_error_exceeded",),
        diagnostics=rejected_diagnostics,
    )
    bundle = NasalTextureObservationBundle(
        trusted=(left,),
        rejected=(rejected,),
        metadata={
            "status": "insufficient_texture_evidence",
            "counts": [1, 0],
        },
    )
    rejected_diagnostics["candidates"].append(3)
    assert bundle.by_side["subject-left"] == (left,)
    assert bundle.by_side["subject-right"] == ()
    assert bundle.metadata["status"] == "insufficient_texture_evidence"
    assert bundle.metadata["counts"] == (1, 0)
    assert bundle.rejected[0].diagnostics["candidates"] == (1, 2)
    encoded = json.dumps(bundle.to_dict(), sort_keys=True)
    assert "insufficient_texture_evidence" in encoded
    assert "epipolar_error_exceeded" in encoded


def test_confidence_maps_separate_highlight_shadow_and_texture() -> None:
    image = np.full((32, 40, 3), 130, dtype=np.uint8)
    semantic = np.ones((32, 40), dtype=np.uint8)
    face = np.ones((32, 40), dtype=np.uint8)
    nostril = np.zeros((32, 40), dtype=np.uint8)

    image[5:10, 5:10] = 255
    image[20:25, 5:10] = 4
    nostril[20:25, 5:10] = 1
    image[10:20, 22:27] = 90
    image[10:20, 27:32] = 190

    maps = build_nasal_texture_confidence_maps(
        image,
        semantic,
        face_mask=face,
        nostril_mask=nostril,
        projected_support=np.ones_like(semantic),
        config=NasalTextureObservationConfig(
            highlight_luma_threshold=245,
            shadow_luma_threshold=18,
            texture_gradient_floor=0.02,
        ),
    )

    assert maps.specular_reject[7, 7]
    assert maps.shadow_reject[22, 7]
    assert maps.final_confidence[7, 7] == 0.0
    assert maps.final_confidence[22, 7] == 0.0
    assert maps.final_confidence[7, 12] == 0.0
    assert maps.final_confidence[22, 12] == 0.0
    assert maps.texture_strength[15, 27] > maps.texture_strength[15, 15]
    assert maps.final_confidence[15, 27] > maps.final_confidence[15, 15]


def test_confidence_maps_reject_outside_semantic_support() -> None:
    image = np.zeros((24, 24, 3), dtype=np.uint8)
    image[:, ::2] = 220
    semantic = np.zeros((24, 24), dtype=np.uint8)
    semantic[6:18, 6:18] = 1
    projected = np.zeros_like(semantic)
    projected[8:16, 8:16] = 1
    maps = build_nasal_texture_confidence_maps(
        image,
        semantic,
        projected_support=projected,
    )

    assert not maps.semantic_support[2, 2]
    assert maps.final_confidence[2, 2] == 0.0
    assert maps.semantic_support[10, 10]
    assert not maps.semantic_support[6, 6]


def test_confidence_maps_reject_moderate_illumination_step() -> None:
    image = np.full((24, 30, 3), 130, dtype=np.uint8)
    image[:, :8] = 40
    support = np.ones((24, 30), dtype=np.uint8)
    maps = build_nasal_texture_confidence_maps(
        image,
        support,
        projected_support=support,
    )
    assert maps.shadow_reject[12, 7]
    assert maps.final_confidence[12, 8] == 0.0

    majority_shadow = np.full((24, 30, 3), 40, dtype=np.uint8)
    majority_shadow[:, 27:] = 130
    maps = build_nasal_texture_confidence_maps(
        majority_shadow,
        support,
        projected_support=support,
    )
    assert maps.shadow_reject[12, 25]
    assert maps.final_confidence[12, 26] == 0.0

    near_highlight = np.full((24, 30, 3), 105, dtype=np.uint8)
    near_highlight[:, :10] = 100
    near_highlight[12, 15] = 244
    maps = build_nasal_texture_confidence_maps(
        near_highlight,
        support,
        projected_support=support,
    )
    assert np.count_nonzero(maps.semantic_support & ~maps.shadow_reject) > 600


def test_confidence_maps_require_projected_support() -> None:
    image = np.full((16, 16, 3), 128, dtype=np.uint8)
    semantic = np.ones((16, 16), dtype=np.uint8)
    with pytest.raises(TypeError, match="projected_support"):
        build_nasal_texture_confidence_maps(image, semantic)


def test_confidence_map_contract_rejects_inconsistent_state() -> None:
    with pytest.raises(ValueError, match="finite"):
        NasalTextureConfidenceMaps(
            semantic_support=np.asarray([[np.nan]]),
            specular_reject=np.asarray([[False]]),
            shadow_reject=np.asarray([[False]]),
            texture_strength=np.asarray([[1.0]]),
            final_confidence=np.asarray([[1.0]]),
        )

    with pytest.raises(ValueError, match="zero outside"):
        NasalTextureConfidenceMaps(
            semantic_support=np.asarray([[False, True]]),
            specular_reject=np.asarray([[False, True]]),
            shadow_reject=np.asarray([[False, False]]),
            texture_strength=np.asarray([[1.0, 1.0]]),
            final_confidence=np.asarray([[1.0, 1.0]]),
        )

    with pytest.raises(ValueError, match="texture_strength"):
        NasalTextureConfidenceMaps(
            semantic_support=np.asarray([[True]]),
            specular_reject=np.asarray([[False]]),
            shadow_reject=np.asarray([[False]]),
            texture_strength=np.asarray([[0.2]]),
            final_confidence=np.asarray([[0.8]]),
        )


def test_config_rejects_fractional_integers_and_impossible_angle() -> None:
    with pytest.raises(ValueError, match="patch_radius_px"):
        NasalTextureObservationConfig(patch_radius_px=2.9)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="highlight_luma_threshold"):
        NasalTextureObservationConfig(
            highlight_luma_threshold=245.9  # type: ignore[arg-type]
        )
    with pytest.raises(ValueError, match="min_ray_angle_deg"):
        NasalTextureObservationConfig(min_ray_angle_deg=181.0)

    config = NasalTextureObservationConfig(
        patch_radius_px=np.int64(4),
        texture_gradient_floor=np.float64(0.02),
    )
    json.dumps(config.to_dict(), sort_keys=True)


def test_covariance_symmetry_uses_zero_relative_tolerance() -> None:
    covariance = np.asarray(
        [[1e6, 1.0, 0.0], [2.0, 1e6, 0.0], [0.0, 0.0, 1e6]],
        dtype=np.float64,
    )
    with pytest.raises(ValueError, match="symmetric"):
        TrustedNasalObservation(
            pair_match=_pair_match(),
            point_reference_m=np.asarray([0.01, -0.02, 0.42]),
            covariance_proxy=covariance,
            depths_m={"front": 0.42, "subject-left": 0.44},
            reprojection_errors_px={"front": 0.4, "subject-left": 0.6},
            ray_angle_deg=14.0,
            weight=0.8,
        )


def test_confidence_map_summary_is_limited_to_semantic_support() -> None:
    support = np.zeros((100, 100), dtype=bool)
    support[45:55, 45:55] = True
    shadow = ~support
    maps = NasalTextureConfidenceMaps(
        semantic_support=support,
        specular_reject=np.zeros_like(support),
        shadow_reject=shadow,
        texture_strength=np.ones((100, 100), dtype=np.float64),
        final_confidence=support.astype(np.float64),
    )
    summary = maps.to_dict()
    assert summary["semantic_support_count"] == 100
    assert summary["shadow_reject_count"] == 0
    assert summary["final_confidence_mean"] == pytest.approx(1.0)


def test_audit_serialization_converts_nonfinite_diagnostics_to_null() -> None:
    rejected = RejectedNasalObservation(
        pair_match=_pair_match(),
        reasons=("no_unique_match",),
        diagnostics={"distance_px": np.inf, "score": np.nan},
    )
    bundle = NasalTextureObservationBundle(
        trusted=(),
        rejected=(rejected,),
        metadata={
            "p90_px": float("inf"),
            "raw_values": np.asarray([1.0, np.inf, np.nan]),
        },
    )
    encoded = json.dumps(bundle.to_dict(), allow_nan=False)
    assert '"distance_px": null' in encoded
    assert '"p90_px": null' in encoded
    assert '"raw_values": [1.0, null, null]' in encoded


def _all_trusted_maps(shape: tuple[int, int]) -> NasalTextureConfidenceMaps:
    support = np.ones(shape, dtype=bool)
    return NasalTextureConfidenceMaps(
        semantic_support=support,
        specular_reject=np.zeros(shape, dtype=bool),
        shadow_reject=np.zeros(shape, dtype=bool),
        texture_strength=np.ones(shape, dtype=np.float64),
        final_confidence=np.ones(shape, dtype=np.float64),
    )


def _slanted_stereo_images() -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(20260731)
    gray = rng.integers(35, 220, size=(64, 80), dtype=np.uint8)
    gray = cv2.GaussianBlur(gray, (0, 0), 0.7)
    front = np.repeat(gray[..., None], 3, axis=2)
    side = np.full_like(front, 128)
    for y in range(front.shape[0]):
        disparity = 6 + int(round(0.04 * y))
        side[y, : front.shape[1] - disparity] = front[y, disparity:]
    return front, side


def test_model_guided_epipolar_match_recovers_slanted_texture() -> None:
    front, side = _slanted_stereo_images()
    y = 30.0
    expected_side = np.asarray([48.0 - (6 + round(0.04 * y)), y])
    seed = NasalEpipolarSeed(
        front_pixel=np.asarray([48.0, y]),
        predicted_side_pixel=expected_side + np.asarray([2.0, 0.0]),
        semantic_region="subject_left_tip_side",
        baseline_vertex_index=7,
    )
    result = match_model_guided_nasal_pair(
        front,
        side,
        side_view="subject-left",
        rig=_synthetic_rig(),
        seeds=(seed,),
        front_confidence=_all_trusted_maps(front.shape[:2]),
        side_confidence=_all_trusted_maps(side.shape[:2]),
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (80, 64), np.eye(3), True
            ),
            "subject-left": NasalCoordinateProvenance(
                "subject-left", (80, 64), (80, 64), np.eye(3), True
            ),
        },
        config=NasalTextureObservationConfig(
            epipolar_search_half_length_px=5.0,
            patch_radius_px=4,
            uniqueness_margin=0.02,
            min_match_confidence=0.45,
        ),
    )
    assert len(result.matches) == 1
    assert np.linalg.norm(result.matches[0].side_pixel - expected_side) <= 1.0
    assert result.matches[0].diagnostics["search_displacement_px"] <= 5.0
    assert result.matches[0].diagnostics["reciprocal_error_px"] <= 1.5


def test_model_guided_epipolar_match_rejects_ambiguous_uniform_patch() -> None:
    image = np.full((64, 80, 3), 128, dtype=np.uint8)
    seed = NasalEpipolarSeed(
        front_pixel=np.asarray([40.0, 30.0]),
        predicted_side_pixel=np.asarray([34.0, 30.0]),
        semantic_region="alar_dome",
        baseline_vertex_index=3,
    )
    result = match_model_guided_nasal_pair(
        image,
        image,
        side_view="subject-right",
        rig=_synthetic_rig(),
        seeds=(seed,),
        front_confidence=_all_trusted_maps(image.shape[:2]),
        side_confidence=_all_trusted_maps(image.shape[:2]),
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (80, 64), np.eye(3), True
            ),
            "subject-right": NasalCoordinateProvenance(
                "subject-right", (80, 64), (80, 64), np.eye(3), True
            ),
        },
    )
    assert result.matches == ()
    assert result.rejected[0].reason in {
        "source_patch_untextured",
        "no_unique_match",
    }


def _synthetic_rig() -> ProfileRig:
    def camera(name: str, view: str, center_x: float) -> Camera:
        rotation = np.eye(3, dtype=np.float64)
        center = np.asarray([center_x, 0.0, 0.0])
        return Camera(
            name=name,
            view=view,
            image_size=(80, 64),
            K=np.asarray(
                [[70.0, 0.0, 40.0], [0.0, 70.0, 32.0], [0.0, 0.0, 1.0]]
            ),
            dist=np.zeros(5),
            R_rig_to_camera=rotation,
            t_rig_to_camera=-rotation @ center,
        )

    return ProfileRig(
        cameras_by_view={
            "left": camera("camera1", "left", -0.08),
            "front": camera("camera2", "front", 0.0),
            "right": camera("camera3", "right", 0.08),
        },
        reference_view="front",
        units="meters",
        calibration_path="synthetic",
        stereo_rms_px={"left": 0.1, "front": 0.0, "right": 0.1},
    )


def test_triangulate_nasal_pair_matches_recovers_metric_point() -> None:
    rig = _synthetic_rig()
    point = np.asarray([0.012, -0.008, 0.48])
    pair = NasalPairMatch(
        side_view="subject-left",
        front_pixel=project_reference_point(point, "front", rig),
        side_pixel=project_reference_point(point, "left", rig),
        confidence=0.92,
        semantic_region="soft_triangle",
        source_matcher="model_guided_gradient_census",
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (80, 64), np.eye(3), True
            ),
            "subject-left": NasalCoordinateProvenance(
                "subject-left", (80, 64), (80, 64), np.eye(3), True
            ),
        },
        diagnostics={"uniqueness_margin": 0.2},
    )
    bundle = triangulate_nasal_pair_matches(
        (pair,),
        rig,
        observed_work_size_by_semantic_view={
            "front": (80, 64),
            "subject-left": (80, 64),
            "subject-right": (80, 64),
        },
    )
    assert len(bundle.trusted) == 1
    assert bundle.trusted[0].point_reference_m == pytest.approx(point, abs=1e-8)
    assert bundle.rejected == ()


def test_model_guided_match_rejects_large_model_to_epipolar_jump() -> None:
    front, side = _slanted_stereo_images()
    seed = NasalEpipolarSeed(
        front_pixel=np.asarray([48.0, 30.0]),
        predicted_side_pixel=np.asarray([42.0, 50.0]),
        semantic_region="alar_dome",
        baseline_vertex_index=10,
    )
    result = match_model_guided_nasal_pair(
        front,
        side,
        side_view="subject-left",
        rig=_synthetic_rig(),
        seeds=(seed,),
        front_confidence=_all_trusted_maps(front.shape[:2]),
        side_confidence=_all_trusted_maps(side.shape[:2]),
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (80, 64), np.eye(3), True
            ),
            "subject-left": NasalCoordinateProvenance(
                "subject-left", (80, 64), (80, 64), np.eye(3), True
            ),
        },
        config=NasalTextureObservationConfig(epipolar_search_half_length_px=5.0),
    )
    assert result.matches == ()
    assert result.rejected[0].reason == "model_epipolar_prior_inconsistent"


def test_model_guided_match_enforces_one_to_one_seed_assignment() -> None:
    front, side = _slanted_stereo_images()
    seed = NasalEpipolarSeed(
        front_pixel=np.asarray([48.0, 30.0]),
        predicted_side_pixel=np.asarray([41.0, 30.0]),
        semantic_region="alar_dome",
        baseline_vertex_index=10,
    )
    result = match_model_guided_nasal_pair(
        front,
        side,
        side_view="subject-left",
        rig=_synthetic_rig(),
        seeds=(seed, seed),
        front_confidence=_all_trusted_maps(front.shape[:2]),
        side_confidence=_all_trusted_maps(side.shape[:2]),
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (80, 64), np.eye(3), True
            ),
            "subject-left": NasalCoordinateProvenance(
                "subject-left", (80, 64), (80, 64), np.eye(3), True
            ),
        },
        config=NasalTextureObservationConfig(
            uniqueness_margin=0.02,
            min_match_confidence=0.45,
        ),
    )
    assert len(result.matches) == 1
    assert any(value.reason == "duplicate_baseline_anchor" for value in result.rejected)


def test_model_guided_match_does_not_reuse_front_texture_patch() -> None:
    front, side = _slanted_stereo_images()
    first = NasalEpipolarSeed(
        front_pixel=np.asarray([48.0, 30.0]),
        predicted_side_pixel=np.asarray([41.0, 30.0]),
        semantic_region="alar_dome",
        baseline_vertex_index=10,
    )
    second = NasalEpipolarSeed(
        front_pixel=np.asarray([48.0, 30.0]),
        predicted_side_pixel=np.asarray([41.0, 30.0]),
        semantic_region="alar_dome",
        baseline_vertex_index=11,
    )
    provenance = {
        "front": NasalCoordinateProvenance(
            "front", (80, 64), (80, 64), np.eye(3), True
        ),
        "subject-left": NasalCoordinateProvenance(
            "subject-left", (80, 64), (80, 64), np.eye(3), True
        ),
    }
    result = match_model_guided_nasal_pair(
        front,
        side,
        side_view="subject-left",
        rig=_synthetic_rig(),
        seeds=(first, second),
        front_confidence=_all_trusted_maps(front.shape[:2]),
        side_confidence=_all_trusted_maps(side.shape[:2]),
        provenance_by_view=provenance,
        config=NasalTextureObservationConfig(
            uniqueness_margin=0.02,
            min_match_confidence=0.45,
        ),
    )
    assert len(result.matches) == 1
    assert any(
        value.reason == "front_pixel_not_one_to_one" for value in result.rejected
    )


def test_triangulation_uses_canonical_source_to_work_scaling() -> None:
    rig = _synthetic_rig()
    point = np.asarray([0.01, -0.005, 0.5])
    transform = np.asarray(
        [[2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    front = project_reference_point(point, "front", rig) * 2.0
    side = project_reference_point(point, "left", rig) * 2.0
    pair = NasalPairMatch(
        side_view="subject-left",
        front_pixel=front,
        side_pixel=side,
        confidence=0.9,
        semantic_region="alar_dome",
        source_matcher="synthetic_affine",
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (160, 128), transform, True
            ),
            "subject-left": NasalCoordinateProvenance(
                "subject-left", (80, 64), (160, 128), transform, True
            ),
        },
        diagnostics={},
    )
    bundle = triangulate_nasal_pair_matches(
        (pair,),
        rig,
        observed_work_size_by_semantic_view={
            "front": (160, 128),
            "subject-left": (160, 128),
            "subject-right": (160, 128),
        },
    )
    assert len(bundle.trusted) == 1
    assert bundle.trusted[0].point_reference_m == pytest.approx(point, abs=1e-8)


def test_model_guided_match_rejects_provenance_that_disagrees_with_image() -> None:
    front, side = _slanted_stereo_images()
    seed = NasalEpipolarSeed(
        front_pixel=np.asarray([48.0, 30.0]),
        predicted_side_pixel=np.asarray([42.0, 30.0]),
        semantic_region="alar_dome",
        baseline_vertex_index=1,
    )
    with pytest.raises(ValueError, match="image size.*provenance"):
        match_model_guided_nasal_pair(
            front,
            side,
            side_view="subject-left",
            rig=_synthetic_rig(),
            seeds=(seed,),
            front_confidence=_all_trusted_maps(front.shape[:2]),
            side_confidence=_all_trusted_maps(side.shape[:2]),
            provenance_by_view={
                "front": NasalCoordinateProvenance(
                    "front", (80, 64), (160, 128), np.diag([2.0, 2.0, 1.0]), True
                ),
                "subject-left": NasalCoordinateProvenance(
                    "subject-left", (80, 64), (160, 128), np.diag([2.0, 2.0, 1.0]), True
                ),
            },
        )


def test_triangulation_rejects_provenance_that_disagrees_with_observed_size() -> None:
    rig = _synthetic_rig()
    pair = NasalPairMatch(
        side_view="subject-left",
        front_pixel=np.asarray([40.0, 32.0]),
        side_pixel=np.asarray([28.0, 32.0]),
        confidence=0.9,
        semantic_region="alar_dome",
        source_matcher="synthetic",
        provenance_by_view={
            "front": NasalCoordinateProvenance(
                "front", (80, 64), (160, 128), np.diag([2.0, 2.0, 1.0]), True
            ),
            "subject-left": NasalCoordinateProvenance(
                "subject-left", (80, 64), (160, 128), np.diag([2.0, 2.0, 1.0]), True
            ),
        },
        diagnostics={},
    )
    with pytest.raises(ValueError, match="work size disagrees"):
        triangulate_nasal_pair_matches(
            (pair,),
            rig,
            observed_work_size_by_semantic_view={
                "front": (80, 64),
                "subject-left": (80, 64),
                "subject-right": (80, 64),
            },
        )


def test_coordinate_provenance_rejects_unverified_crop_or_translation() -> None:
    with pytest.raises(ValueError, match="canonical full-frame resize"):
        NasalCoordinateProvenance(
            "front",
            (80, 64),
            (80, 64),
            np.asarray(
                [[1.0, 0.0, 5.0], [0.0, 1.0, 3.0], [0.0, 0.0, 1.0]],
                dtype=np.float64,
            ),
            True,
        )

from __future__ import annotations

import json

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.nasal_texture_observations import NasalCoordinateProvenance
from src.geometry.nasal_view_registration import (
    BaselineNasalRegistrationSurface,
    FixedNasalViewRegistration,
    NasalViewRegistrationConfig,
    ViewRegistrationSample,
    estimate_fixed_nasal_view_offsets,
)
from src.geometry.profile_triangulation import ProfileRig


VIEWS = ("front", "subject-left", "subject-right")


def _registration_rig() -> ProfileRig:
    def camera(name: str, view: str, center_x: float) -> Camera:
        rotation = np.eye(3, dtype=np.float64)
        center = np.asarray([center_x, 0.0, 0.0], dtype=np.float64)
        return Camera(
            name=name,
            view=view,
            image_size=(100, 100),
            K=np.eye(3, dtype=np.float64),
            dist=np.zeros(5, dtype=np.float64),
            R_rig_to_camera=rotation,
            t_rig_to_camera=-rotation @ center,
        )

    return ProfileRig(
        cameras_by_view={
            "left": camera("camera1", "left", -0.1),
            "front": camera("camera2", "front", 0.0),
            "right": camera("camera3", "right", 0.1),
        },
        reference_view="front",
        units="meters",
        calibration_path="synthetic",
        stereo_rms_px={"left": 0.1, "front": 0.0, "right": 0.1},
    )


def _registration_provenance() -> dict[str, NasalCoordinateProvenance]:
    return {
        view: NasalCoordinateProvenance(
            semantic_view=view,
            source_size=(100, 100),
            work_size=(100, 100),
            source_to_work=np.eye(3, dtype=np.float64),
            undistorted=True,
        )
        for view in VIEWS
    }


def _baseline_surface() -> BaselineNasalRegistrationSurface:
    vertices = np.asarray(
        [
            [-1.0, 0.0, 0.0], [0.0, 0.0, 0.0], [-0.5, 1.0, 0.0],
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, 1.0, 0.0],
            [-1.0, -1.0, 0.0], [0.0, -1.0, 0.0], [-0.5, 0.0, 0.0],
            [0.0, -1.0, 0.0], [1.0, -1.0, 0.0], [0.5, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    faces = np.asarray(
        [[0, 1, 2], [3, 4, 5], [6, 7, 8], [9, 10, 11]],
        dtype=np.int64,
    )
    return BaselineNasalRegistrationSurface(
        vertices=vertices,
        faces=faces,
        protected_face_indices_by_region={
            "bridge": np.asarray([0]),
            "peri_nasal_skin": np.asarray([1]),
            "subject_left_peri_nasal": np.asarray([2]),
            "subject_right_peri_nasal": np.asarray([3]),
        },
        rig=_registration_rig(),
        provenance_by_view=_registration_provenance(),
        model_to_front_rotation=np.eye(3, dtype=np.float64),
        model_to_front_translation=np.asarray([10.45, 9.5, 1.0]),
    )


def _projected_for_region(region: str) -> np.ndarray:
    return np.asarray([10.0, 10.0]) if region == "bridge" else np.asarray([11.0, 10.0])


def test_registration_surface_derives_projection_from_fixed_rig() -> None:
    surface = _baseline_surface()
    front = surface.projection_matrices_by_view["front"]
    left = surface.projection_matrices_by_view["subject-left"]
    assert front == pytest.approx(
        np.asarray(
            [[1.0, 0.0, 0.0, 10.45], [0.0, 1.0, 0.0, 9.5], [0.0, 0.0, 1.0, 1.0]]
        )
    )
    assert left[0, 3] == pytest.approx(front[0, 3] + 0.1)
    with pytest.raises(TypeError, match="projection_matrices_by_view"):
        BaselineNasalRegistrationSurface(
            vertices=surface.vertices,
            faces=surface.faces,
            protected_face_indices_by_region=surface.protected_face_indices_by_region,
            rig=surface.rig,
            provenance_by_view=surface.provenance_by_view,
            model_to_front_rotation=surface.model_to_front_rotation,
            model_to_front_translation=surface.model_to_front_translation,
            projection_matrices_by_view={view: np.zeros((3, 4)) for view in VIEWS},
        )


def test_registration_surface_rejects_duplicate_camera_mapping() -> None:
    surface = _baseline_surface()
    with pytest.raises(ValueError, match="one-to-one"):
        BaselineNasalRegistrationSurface(
            vertices=surface.vertices,
            faces=surface.faces,
            protected_face_indices_by_region=surface.protected_face_indices_by_region,
            rig=surface.rig,
            provenance_by_view=surface.provenance_by_view,
            model_to_front_rotation=surface.model_to_front_rotation,
            model_to_front_translation=surface.model_to_front_translation,
            rig_view_by_semantic={view: "front" for view in VIEWS},
        )


def _samples(
    offsets: dict[str, np.ndarray],
    *,
    count: int = 24,
    outliers: bool = False,
) -> tuple[ViewRegistrationSample, ...]:
    rng = np.random.default_rng(20260731)
    result = []
    surface = _baseline_surface()
    for view_index, view in enumerate(VIEWS):
        for index in range(count):
            region = "bridge" if index % 2 == 0 else "peri_nasal_skin"
            face_index = 0 if index % 2 == 0 else 1
            barycentric = np.asarray(
                [
                    0.12 + 0.012 * (index % 8),
                    0.24 + 0.008 * (index // 8),
                    0.0,
                ],
                dtype=np.float64,
            )
            barycentric[2] = 1.0 - barycentric[0] - barycentric[1]
            model_point = np.sum(
                surface.vertices[surface.faces[face_index]] * barycentric[:, None],
                axis=0,
            )
            projected_h = surface.projection_matrices_by_view[view] @ np.append(
                model_point,
                1.0,
            )
            projected = projected_h[:2] / projected_h[2]
            noise = rng.normal(0.0, 0.035, size=2)
            observed = projected + offsets[view] + noise
            if outliers and index in (1, 7):
                observed += np.asarray([18.0, -22.0])
            result.append(
                ViewRegistrationSample(
                    semantic_view=view,
                    projected_pixel=projected,
                    observed_pixel=observed,
                    confidence=0.95,
                    semantic_region=region,
                    source="protected_surface_correspondence",
                    baseline_face_index=face_index,
                    baseline_bary_coords=barycentric,
                    matched_point_3d=model_point + np.asarray([0.0, 0.0, 0.001]),
                )
            )
    return tuple(result)


def test_fixed_registration_recovers_robust_zero_mean_offsets() -> None:
    expected = {
        "front": np.asarray([1.0, -0.5]),
        "subject-left": np.asarray([-0.5, 0.25]),
        "subject-right": np.asarray([-0.5, 0.25]),
    }
    result = estimate_fixed_nasal_view_offsets(
        _samples(expected, outliers=True),
        _baseline_surface(),
        NasalViewRegistrationConfig(min_samples_per_view=8),
    )

    assert isinstance(result, FixedNasalViewRegistration)
    assert result.all_views_accepted
    for view in VIEWS:
        assert np.linalg.norm(result.offsets_by_view[view] - expected[view]) < 0.25
        assert result.sample_counts_by_view[view] == 24
        assert result.residual_p90_px_by_view[view] < 0.2
        assert result.inlier_counts_by_view[view] >= 20
        assert result.semantic_region_counts_by_view[view] == 2
        assert result.residual_quantiles_px_by_view[view].shape == (3,)
        assert result.offset_confidence_interval95_px_by_view[view].shape == (2, 2)
    assert np.linalg.norm(sum(result.offsets_by_view.values())) < 1e-8
    with pytest.raises(ValueError):
        result.offsets_by_view["front"][0] = 99.0
    json.dumps(result.to_dict(), allow_nan=False)


def test_registration_falls_back_to_zero_for_low_support_view() -> None:
    expected = {view: np.zeros(2, dtype=np.float64) for view in VIEWS}
    samples = tuple(
        sample
        for sample in _samples(expected, count=8)
            if not (
                sample.semantic_view == "subject-right"
                and sample.projected_pixel[0] > 10.5
            )
    )
    result = estimate_fixed_nasal_view_offsets(
        samples,
        _baseline_surface(),
        NasalViewRegistrationConfig(min_samples_per_view=6),
    )

    assert not result.accepted_by_view["subject-right"]
    assert np.array_equal(
        result.offsets_by_view["subject-right"],
        np.zeros(2, dtype=np.float64),
    )
    assert result.confidence_by_view["subject-right"] == 0.0
    assert not result.all_views_accepted
    assert np.linalg.norm(sum(result.offsets_by_view.values())) < 1e-8


def test_registration_ignores_unprotected_regions_and_low_confidence() -> None:
    valid = ViewRegistrationSample(
        semantic_view="front",
        projected_pixel=np.asarray([10.0, 10.0]),
        observed_pixel=np.asarray([11.0, 10.0]),
        confidence=0.9,
        semantic_region="bridge",
        source="protected_surface_correspondence",
        baseline_face_index=0,
        baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
        matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
    )
    cheek = ViewRegistrationSample(
        semantic_view="front",
        projected_pixel=np.asarray([10.0, 10.0]),
        observed_pixel=np.asarray([99.0, 99.0]),
        confidence=1.0,
        semantic_region="cheek",
        source="untrusted_region",
        baseline_face_index=0,
        baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
        matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
    )
    low = ViewRegistrationSample(
        semantic_view="front",
        projected_pixel=np.asarray([10.0, 10.0]),
        observed_pixel=np.asarray([99.0, 99.0]),
        confidence=0.1,
        semantic_region="bridge",
        source="low_confidence",
        baseline_face_index=0,
        baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
        matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
    )
    result = estimate_fixed_nasal_view_offsets(
        (valid, cheek, low),
        _baseline_surface(),
        NasalViewRegistrationConfig(
            min_samples_per_view=1,
            min_semantic_regions_per_view=1,
            min_sample_confidence=0.5,
        ),
    )
    assert result.sample_counts_by_view["front"] == 1
    assert np.array_equal(result.offsets_by_view["front"], np.zeros(2))


def test_registration_sample_rejects_invalid_pixels_and_view() -> None:
    with pytest.raises(ValueError, match="semantic_view"):
        ViewRegistrationSample(
            semantic_view="camera2",
            projected_pixel=np.asarray([10.0, 10.0]),
            observed_pixel=np.asarray([11.0, 10.0]),
            confidence=0.9,
            semantic_region="bridge",
            source="test",
            baseline_face_index=0,
            baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
            matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
        )
    with pytest.raises(ValueError, match="projected_pixel"):
        ViewRegistrationSample(
            semantic_view="front",
            projected_pixel=np.asarray([np.nan, 10.0]),
            observed_pixel=np.asarray([11.0, 10.0]),
            confidence=0.9,
            semantic_region="bridge",
            source="test",
            baseline_face_index=0,
            baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
            matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
        )


def test_registration_rejects_unstable_or_unverified_estimates() -> None:
    unstable = []
    for index in range(12):
        residual = np.asarray([3.0, -3.0]) if index % 2 else np.asarray([-3.0, 3.0])
        unstable.append(
            ViewRegistrationSample(
                semantic_view="front",
                projected_pixel=np.asarray([50.0 + index, 50.0]),
                observed_pixel=np.asarray([50.0 + index, 50.0]) + residual,
                confidence=0.9,
                semantic_region="bridge",
                source="protected_surface_correspondence",
                baseline_face_index=0,
                baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
                matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
            )
        )
    unverified = ViewRegistrationSample(
        semantic_view="subject-left",
        projected_pixel=np.asarray([50.0, 50.0]),
        observed_pixel=np.asarray([51.0, 50.0]),
        confidence=0.9,
        semantic_region="bridge",
        source="caller_claimed_bridge",
        baseline_face_index=0,
        baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
        matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
    )
    result = estimate_fixed_nasal_view_offsets(
        tuple(unstable) + (unverified,),
        _baseline_surface(),
        NasalViewRegistrationConfig(
            min_samples_per_view=1,
            min_semantic_regions_per_view=1,
            max_residual_p90_px=1.0,
        ),
    )
    assert not result.accepted_by_view["front"]
    assert np.array_equal(result.offsets_by_view["front"], np.zeros(2))
    assert result.sample_counts_by_view["subject-left"] == 0


def test_registration_rejects_faces_outside_verified_region() -> None:
    forged = ViewRegistrationSample(
        semantic_view="front",
        projected_pixel=np.asarray([10.0, 10.0]),
        observed_pixel=np.asarray([11.0, 10.0]),
        confidence=0.95,
        semantic_region="bridge",
        source="protected_surface_correspondence",
        baseline_face_index=1,
        baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
        matched_point_3d=np.asarray([0.55, 0.5, 0.001]),
    )
    out_of_range = ViewRegistrationSample(
        semantic_view="front",
        projected_pixel=np.asarray([10.0, 10.0]),
        observed_pixel=np.asarray([11.0, 10.0]),
        confidence=0.95,
        semantic_region="bridge",
        source="protected_surface_correspondence",
        baseline_face_index=10**12,
        baseline_bary_coords=np.asarray([0.2, 0.3, 0.5]),
        matched_point_3d=np.asarray([-0.45, 0.5, 0.001]),
    )
    result = estimate_fixed_nasal_view_offsets(
        (forged, out_of_range),
        _baseline_surface(),
        NasalViewRegistrationConfig(min_samples_per_view=1),
    )
    assert result.sample_counts_by_view["front"] == 0
    assert not result.accepted_by_view["front"]


def test_registration_deduplicates_identical_surface_anchors() -> None:
    sample = _samples({view: np.zeros(2) for view in VIEWS}, count=1)[0]
    result = estimate_fixed_nasal_view_offsets(
        tuple(sample for _ in range(6)),
        _baseline_surface(),
        NasalViewRegistrationConfig(
            min_samples_per_view=6,
            min_semantic_regions_per_view=1,
        ),
    )
    assert result.sample_counts_by_view["front"] == 1
    assert not result.accepted_by_view["front"]


def test_registration_rechecks_offset_bound_after_gauge() -> None:
    offsets = {
        "front": np.asarray([4.0, 0.0]),
        "subject-left": np.asarray([-3.0, 0.0]),
        "subject-right": np.asarray([-3.0, 0.0]),
    }
    result = estimate_fixed_nasal_view_offsets(
        _samples(offsets),
        _baseline_surface(),
        NasalViewRegistrationConfig(
            min_samples_per_view=8,
            max_offset_px=4.1,
            max_residual_p90_px=2.0,
        ),
    )
    for view in VIEWS:
        assert np.linalg.norm(result.offsets_by_view[view]) <= 4.1 + 1e-9
    assert not result.accepted_by_view["front"]


def test_registration_requires_enough_robust_inliers() -> None:
    samples = list(_samples({view: np.zeros(2) for view in VIEWS}, count=6))
    front_indices = [i for i, sample in enumerate(samples) if sample.semantic_view == "front"]
    sample = samples[front_indices[-1]]
    samples[front_indices[-1]] = ViewRegistrationSample(
        **{**sample.__dict__, "observed_pixel": sample.observed_pixel + np.asarray([20.0, -20.0])}
    )
    result = estimate_fixed_nasal_view_offsets(
        tuple(samples),
        _baseline_surface(),
        NasalViewRegistrationConfig(min_samples_per_view=6),
    )
    assert result.sample_counts_by_view["front"] == 6
    assert result.inlier_counts_by_view["front"] == 5
    assert not result.accepted_by_view["front"]


def test_registration_rejects_pixel_not_projected_from_same_surface_point() -> None:
    sample = _samples({view: np.zeros(2) for view in VIEWS}, count=1)[0]
    mismatched = ViewRegistrationSample(
        **{**sample.__dict__, "projected_pixel": sample.projected_pixel + np.asarray([3.0, 0.0])}
    )
    result = estimate_fixed_nasal_view_offsets(
        (mismatched,),
        _baseline_surface(),
        NasalViewRegistrationConfig(min_samples_per_view=1),
    )
    assert result.sample_counts_by_view["front"] == 0
    assert not result.accepted_by_view["front"]

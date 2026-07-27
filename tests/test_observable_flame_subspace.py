from __future__ import annotations

import json
from dataclasses import FrozenInstanceError

import numpy as np
import pytest

from src.geometry.observable_flame_subspace import (
    OBSERVABLE_FLAME_VIEW_NAMES,
    ObservableFlameSubspaceConfig,
    ProjectionView,
    build_observable_flame_subspace,
    perspective_projection_jacobian,
)


def _rotation_y(angle_degrees: float) -> np.ndarray:
    angle = np.deg2rad(angle_degrees)
    return np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ],
        dtype=np.float64,
    )


def _view(
    name: str,
    *,
    angle: float = 0.0,
    focal: float = 800.0,
    translation=(0.0, 0.0, 0.0),
) -> ProjectionView:
    return ProjectionView(
        name=name,
        K=np.array(
            [
                [focal, 0.0, 320.0],
                [0.0, focal * 1.05, 240.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        ),
        R_model_to_camera=_rotation_y(angle),
        t_model_to_camera=np.asarray(translation, dtype=np.float64),
    )


def _views(*, focal_scale: float = 1.0):
    return (
        _view("subject-right", angle=-24.0, focal=760.0 * focal_scale),
        _view("front", focal=900.0 * focal_scale),
        _view("subject-left", angle=27.0, focal=820.0 * focal_scale),
    )


def test_real_like_near_rotation_is_projected_to_proper_so3():
    angle = np.deg2rad(12.0)
    exact = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    target_determinant = 0.99999968
    first_scale = np.sqrt(1.0 - 4.41e-7)
    real_like = (
        np.diag([first_scale, target_determinant / first_scale, 1.0])
        @ exact
    )

    error = float(np.max(np.abs(real_like @ real_like.T - np.eye(3))))
    assert np.linalg.det(real_like) == pytest.approx(
        target_determinant,
        abs=1e-12,
    )
    assert error == pytest.approx(4.41e-7, abs=1e-12)
    view = ProjectionView(
        "front",
        _view("front").K,
        real_like,
        np.zeros(3),
    )

    np.testing.assert_allclose(
        view.R_model_to_camera @ view.R_model_to_camera.T,
        np.eye(3),
        atol=1e-12,
    )
    assert np.linalg.det(view.R_model_to_camera) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(view.R_model_to_camera - real_like) < 1e-5

    shear = np.eye(3)
    shear[0, 1] = 0.01
    with pytest.raises(ValueError, match="rotation.*rigid"):
        ProjectionView("front", _view("front").K, shear, np.zeros(3))
    with pytest.raises(ValueError, match="determinant|reflection"):
        ProjectionView(
            "front",
            _view("front").K,
            np.diag([-1.0, 1.0, 1.0]),
            np.zeros(3),
        )


def _problem(vertex_count: int = 10, mode_count: int = 4):
    x = np.linspace(-0.35, 0.42, vertex_count)
    y = np.linspace(-0.22, 0.31, vertex_count)
    z = np.linspace(3.8, 4.4, vertex_count)
    vertices = np.column_stack((x, y, z))
    shape_basis = np.zeros((vertex_count, 3, mode_count), dtype=np.float64)
    support_mask = np.zeros(vertex_count, dtype=bool)
    support_mask[:3] = True
    protected_mask = np.zeros(vertex_count, dtype=bool)
    protected_mask[3:5] = True
    return vertices, shape_basis, support_mask, protected_mask


def _config(**overrides) -> ObservableFlameSubspaceConfig:
    values = {
        "min_nasal_response_ratio": 0.60,
        "max_protected_to_nasal_energy_ratio": 0.50,
        "relative_nasal_energy_floor": 0.02,
        "relative_singular_value_threshold": 1e-7,
        "max_rank": 12,
    }
    values.update(overrides)
    return ObservableFlameSubspaceConfig(**values)


def _build(
    shape_basis,
    *,
    config=None,
    vertices=None,
    support=None,
    protected=None,
    views=None,
):
    vertex_count = (
        shape_basis.shape[0]
        if shape_basis.ndim == 3
        else shape_basis.shape[0] // 3
    )
    mode_count = (
        shape_basis.shape[2]
        if shape_basis.ndim == 3
        else shape_basis.shape[1]
    )
    defaults = _problem(vertex_count, mode_count)
    return build_observable_flame_subspace(
        defaults[0] if vertices is None else vertices,
        shape_basis,
        defaults[2] if support is None else support,
        defaults[3] if protected is None else protected,
        _views() if views is None else views,
        config=_config() if config is None else config,
    )


def test_nose_inactive_modes_are_excluded_and_synthetic_nasal_modes_retained():
    _vertices, shape_basis, _support, _protected = _problem()
    shape_basis[:3, 0, 0] = [1.0, 0.8, 1.1]
    shape_basis[5:, 1, 1] = 1.0
    shape_basis[:3, 1, 2] = [0.7, 1.0, 0.9]
    shape_basis[:, 2, 3] = 0.0

    result = _build(shape_basis)

    np.testing.assert_array_equal(result.candidate_mode_indices, [0, 2])
    assert result.screening_pass_mask.tolist() == [True, False, True, False]
    assert result.retained_rank == 2
    assert result.report_data["status"] == "ok"


def test_outside_and_protected_penalties_are_independent():
    _vertices, shape_basis, _support, _protected = _problem(mode_count=3)
    shape_basis[:3, 0, :] = 1.0
    shape_basis[5:, 0, 0] = 1.0
    shape_basis[3:5, 0, 1] = 1.0

    result = _build(shape_basis, config=_config(relative_nasal_energy_floor=0.0))

    assert result.nasal_response_ratio[0] == pytest.approx(0.5)
    assert result.protected_to_nasal_energy_ratio[0] == pytest.approx(0.0)
    assert result.nasal_response_ratio[1] == pytest.approx(1.0)
    assert result.protected_to_nasal_energy_ratio[1] == pytest.approx(1.0)
    assert result.screening_pass_mask.tolist() == [False, False, True]


def test_per_vertex_energy_normalization_avoids_mask_size_bias():
    config = _config(
        min_nasal_response_ratio=0.0,
        max_protected_to_nasal_energy_ratio=2.0,
        relative_nasal_energy_floor=0.0,
    )
    ratios = []
    for vertex_count, outside_start in ((8, 5), (20, 5)):
        vertices, shape_basis, support, protected = _problem(vertex_count, 1)
        shape_basis[support, 0, 0] = 2.0
        shape_basis[outside_start:, 0, 0] = 1.0
        result = _build(
            shape_basis,
            config=config,
            vertices=vertices,
            support=support,
            protected=protected,
        )
        ratios.append(result.nasal_response_ratio[0])

    assert ratios[0] == pytest.approx(4.0 / 5.0)
    assert ratios[1] == pytest.approx(ratios[0])


def test_screening_metrics_are_mean_per_vertex_squared_displacement_energy():
    vertices, shape_basis, support, protected = _problem(10, 1)
    shape_basis[support, :, 0] = np.array([3.0, 4.0, 0.0])
    shape_basis[protected, :, 0] = np.array([0.0, 3.0, 4.0])
    shape_basis[~(support | protected), :, 0] = np.array([0.0, 0.0, 2.0])
    result = _build(
        shape_basis,
        config=_config(
            min_nasal_response_ratio=0.0,
            max_protected_to_nasal_energy_ratio=2.0,
            relative_nasal_energy_floor=0.0,
        ),
        vertices=vertices,
        support=support,
        protected=protected,
    )

    assert result.nasal_mean_squared_energy[0] == pytest.approx(25.0)
    assert result.outside_mean_squared_energy[0] == pytest.approx(4.0)
    assert result.protected_mean_squared_energy[0] == pytest.approx(25.0)
    assert result.nasal_response_ratio[0] == pytest.approx(25.0 / 29.0)
    assert result.outside_to_nasal_energy_ratio[0] == pytest.approx(4.0 / 25.0)
    assert result.protected_to_nasal_energy_ratio[0] == pytest.approx(1.0)
    assert result.relative_nasal_energy[0] == pytest.approx(1.0)
    screening_report = result.report_data["screening"]
    assert screening_report["energy_normalization"] == (
        "mean per-vertex squared displacement"
    )
    assert "nasal_rms" not in screening_report


def test_tiny_absolute_energy_high_ratio_mode_is_excluded_by_relative_floor():
    _vertices, shape_basis, _support, _protected = _problem(mode_count=2)
    shape_basis[:3, 0, 0] = 1.0
    shape_basis[:3, 1, 1] = 1e-5

    result = _build(shape_basis, config=_config(relative_nasal_energy_floor=0.01))

    np.testing.assert_array_equal(result.candidate_mode_indices, [0])
    assert result.nasal_response_ratio[1] == pytest.approx(1.0)
    assert result.relative_nasal_energy[1] == pytest.approx(1e-10)


def _project(vertices: np.ndarray, view: ProjectionView) -> np.ndarray:
    camera = vertices @ view.R_model_to_camera.T + view.t_model_to_camera
    homogeneous = camera @ view.K.T
    return homogeneous[:, :2] / homogeneous[:, 2, None]


def test_analytic_jacobian_matches_independent_central_difference_all_views():
    vertices, shape_basis, support, _protected = _problem(mode_count=3)
    shape_basis[:3, :, 0] = np.array([0.2, -0.1, 0.05])
    shape_basis[:3, :, 1] = np.array([-0.08, 0.14, 0.03])
    shape_basis[:3, :, 2] = np.array([0.04, 0.02, -0.12])
    support_vertices = vertices[support]
    support_basis = shape_basis[support]
    step = 1e-6

    for view in _views():
        analytic = perspective_projection_jacobian(
            support_vertices,
            support_basis,
            view,
        )
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            analytic.setflags(write=True)
        numeric = np.empty_like(analytic)
        for mode in range(shape_basis.shape[2]):
            delta = support_basis[:, :, mode] * step
            positive = _project(support_vertices + delta, view).reshape(-1)
            negative = _project(support_vertices - delta, view).reshape(-1)
            numeric[:, mode] = (positive - negative) / (2.0 * step)
        np.testing.assert_allclose(analytic, numeric, rtol=2e-7, atol=2e-7)


def test_synthetic_observable_modes_are_retained_by_svd():
    _vertices, shape_basis, _support, _protected = _problem(mode_count=3)
    shape_basis[:3, 0, 0] = [1.0, 0.4, -0.2]
    shape_basis[:3, 1, 1] = [-0.3, 0.8, 0.2]
    shape_basis[:3, 2, 2] = [0.2, -0.15, 0.35]

    result = _build(shape_basis)

    assert result.retained_rank == 3
    assert np.all(result.singular_values[:3] > 0.0)
    assert all(item.rank == 3 for item in result.per_view_metadata)


def test_rank_deficient_duplicate_modes_reduce_rank_without_large_or_nonfinite_values():
    _vertices, shape_basis, _support, _protected = _problem(mode_count=3)
    shape_basis[:3, 0, 0] = [1.0, 0.5, -0.25]
    shape_basis[:, :, 1] = shape_basis[:, :, 0]
    shape_basis[:3, 1, 2] = [0.2, 0.8, -0.1]

    first = _build(shape_basis)
    second = _build(shape_basis)

    assert first.retained_rank == 2
    np.testing.assert_array_equal(first.coefficient_basis, second.coefficient_basis)
    assert np.isfinite(first.coefficient_basis).all()
    assert np.max(np.abs(first.coefficient_basis)) <= 1.0
    assert np.isfinite(first.vertex_basis).all()


def test_coefficient_basis_is_orthonormal_and_vertex_basis_is_exact_contraction():
    _vertices, shape_basis, _support, _protected = _problem(mode_count=4)
    shape_basis[:3, 0, 0] = [1.0, 0.5, -0.2]
    shape_basis[:3, 1, 1] = [0.1, 0.8, -0.4]
    shape_basis[:3, 2, 2] = [0.3, -0.2, 0.7]
    shape_basis[:3, :, 3] = np.array([0.2, -0.15, 0.1])

    result = _build(shape_basis)
    expected = np.einsum(
        "vcs,sr->vcr",
        shape_basis,
        result.coefficient_basis,
        optimize=True,
    )

    np.testing.assert_allclose(
        result.coefficient_basis.T @ result.coefficient_basis,
        np.eye(result.retained_rank),
        atol=1e-12,
    )
    np.testing.assert_allclose(result.vertex_basis, expected, atol=1e-14)


def test_camera_geometry_changes_observability_without_changing_semantic_mapping():
    vertices, shape_basis, support, protected = _problem(mode_count=2)
    shape_basis[:3, 0, 0] = [1.0, 0.4, -0.1]
    shape_basis[:3, 2, 1] = [0.5, -0.3, 0.7]
    baseline = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
    )
    changed_views = (
        _view("subject-right", angle=-58.0),
        _view("subject-left", angle=51.0),
        _view("front", angle=8.0),
    )
    changed = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
        views=changed_views,
    )

    assert baseline.view_names == changed.view_names == OBSERVABLE_FLAME_VIEW_NAMES
    assert [item.name for item in changed.per_view_metadata] == list(
        OBSERVABLE_FLAME_VIEW_NAMES
    )
    assert not np.allclose(baseline.singular_values, changed.singular_values)


def test_same_config_rules_hold_for_scaled_subjects_and_camera_focals():
    vertices, shape_basis, support, protected = _problem(mode_count=3)
    shape_basis[:3, 0, 0] = 1.0
    shape_basis[:3, 1, 1] = 0.2
    shape_basis[5:, 1, 1] = 1.0
    shape_basis[:3, 2, 2] = 0.4
    config = _config()
    first = _build(
        shape_basis,
        config=config,
        vertices=vertices,
        support=support,
        protected=protected,
        views=_views(focal_scale=1.0),
    )
    scaled_views = tuple(
        ProjectionView(
            name=view.name,
            K=view.K * np.array([[3.0, 3.0, 3.0], [3.0, 3.0, 3.0], [1.0, 1.0, 1.0]]),
            R_model_to_camera=view.R_model_to_camera,
            t_model_to_camera=view.t_model_to_camera * 2.5,
        )
        for view in _views()
    )
    second = _build(
        shape_basis * 2.5,
        config=config,
        vertices=vertices * 2.5,
        support=support,
        protected=protected,
        views=scaled_views,
    )

    assert first.config is config
    assert second.config is config
    np.testing.assert_array_equal(
        first.candidate_mode_indices,
        second.candidate_mode_indices,
    )
    np.testing.assert_allclose(
        first.singular_values,
        second.singular_values,
        atol=1e-12,
    )
    assert first.report_data["config"] == second.report_data["config"]
    assert first.report_data["view_validation"] == {
        "rotation_orthogonality_tolerance": 1e-5,
        "rotation_determinant_tolerance": 1e-5,
        "accepted_rotation_handling": "nearest proper SO(3) via SVD",
        "intrinsic_final_row_tolerance": 1e-10,
    }
    assert "dataset" not in json.dumps(first.to_report_data()).lower()


def test_anisotropic_intrinsics_and_skew_do_not_change_observability():
    vertices, shape_basis, support, protected = _problem(mode_count=3)
    shape_basis[:3, 0, 0] = [1.0, 0.4, -0.2]
    shape_basis[:3, 1, 1] = [-0.3, 0.8, 0.2]
    shape_basis[:3, 2, 2] = [0.2, -0.15, 0.35]
    intrinsic_a = np.array(
        [
            [700.0, 80.0, 320.0],
            [0.0, 1100.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    intrinsic_b = np.array(
        [
            [1700.0, -240.0, 510.0],
            [0.0, 360.0, 190.0],
            [0.0, 0.0, 1.0],
        ]
    )
    geometry = _views()
    views_a = tuple(
        ProjectionView(
            view.name,
            intrinsic_a,
            view.R_model_to_camera,
            view.t_model_to_camera,
        )
        for view in geometry
    )
    views_b = tuple(
        ProjectionView(
            view.name,
            intrinsic_b,
            view.R_model_to_camera,
            view.t_model_to_camera,
        )
        for view in geometry
    )

    raw_a = perspective_projection_jacobian(
        vertices[support],
        shape_basis[support],
        views_a[0],
    )
    raw_b = perspective_projection_jacobian(
        vertices[support],
        shape_basis[support],
        views_b[0],
    )
    assert not np.allclose(raw_a, raw_b)
    result_a = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
        views=views_a,
    )
    result_b = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
        views=views_b,
    )

    np.testing.assert_allclose(
        result_a.singular_values,
        result_b.singular_values,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        result_a.coefficient_basis,
        result_b.coefficient_basis,
        atol=1e-12,
    )
    expected_a = tuple(
        tuple(float(item) for item in row)
        for row in np.linalg.inv(intrinsic_a[:2, :2])
    )
    assert result_a.per_view_metadata[0].intrinsic_normalization_matrix == (
        expected_a
    )
    assert (
        "inverse 2x2 intrinsic linear block"
        in result_a.report_data["row_balancing"]
    )


def test_no_screened_modes_returns_diagnostic_zero_rank_without_fallback():
    _vertices, shape_basis, _support, _protected = _problem(mode_count=4)

    result = _build(shape_basis)

    assert result.retained_rank == 0
    assert result.coefficient_basis.shape == (4, 0)
    assert result.vertex_basis.shape == (10, 3, 0)
    assert result.candidate_mode_indices.size == 0
    assert result.report_data["status"] == "no_candidate_modes"


def test_zero_projection_rank_returns_diagnostic_zero_rank():
    vertices, shape_basis, support, protected = _problem(mode_count=1)
    shape_basis[:3, 2, 0] = vertices[:3, 2]
    shape_basis[:3, 0, 0] = vertices[:3, 0]
    shape_basis[:3, 1, 0] = vertices[:3, 1]
    shared_view = _view("front")
    views = (
        shared_view,
        ProjectionView(
            "subject-left",
            shared_view.K,
            shared_view.R_model_to_camera,
            shared_view.t_model_to_camera,
        ),
        ProjectionView(
            "subject-right",
            shared_view.K,
            shared_view.R_model_to_camera,
            shared_view.t_model_to_camera,
        ),
    )

    result = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
        views=views,
    )

    assert result.candidate_mode_indices.tolist() == [0]
    assert result.retained_rank == 0
    assert result.report_data["status"] == "no_observable_rank"


@pytest.mark.parametrize(
    ("field", "replacement", "match"),
    [
        ("vertices", np.zeros((4, 2)), "vertices"),
        ("shape_basis", np.zeros((10, 2, 3)), "shape_basis"),
        ("support", np.zeros(9, dtype=bool), "support_mask"),
        ("support", np.arange(10), "support_mask.*boolean"),
        ("protected", np.zeros(9, dtype=bool), "protected_mask"),
    ],
)
def test_invalid_arrays_and_masks_fail_clearly(field, replacement, match):
    vertices, shape_basis, support, protected = _problem(mode_count=3)
    values = {
        "vertices": vertices,
        "shape_basis": shape_basis,
        "support": support,
        "protected": protected,
    }
    values[field] = replacement

    with pytest.raises(ValueError, match=match):
        _build(
            values["shape_basis"],
            vertices=values["vertices"],
            support=values["support"],
            protected=values["protected"],
        )


def test_flattened_shape_basis_is_supported_and_mask_overlap_is_rejected():
    vertices, shape_basis, support, protected = _problem(mode_count=2)
    shape_basis[:3, 0, 0] = 1.0
    shape_basis[:3, 1, 1] = 1.0
    volumetric = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
    )
    flattened = _build(
        shape_basis.reshape(len(vertices) * 3, 2),
        vertices=vertices,
        support=support,
        protected=protected,
    )

    np.testing.assert_array_equal(volumetric.vertex_basis, flattened.vertex_basis)
    bad_protected = protected.copy()
    bad_protected[0] = True
    with pytest.raises(ValueError, match="disjoint"):
        _build(
            shape_basis,
            vertices=vertices,
            support=support,
            protected=bad_protected,
        )


def test_invalid_views_intrinsics_extrinsics_depth_and_config_fail_clearly():
    vertices, shape_basis, support, protected = _problem(mode_count=1)
    shape_basis[:3, 0, 0] = 1.0
    with pytest.raises(ValueError, match="exactly"):
        _build(shape_basis, views=_views()[:2])
    with pytest.raises(ValueError, match="view name"):
        ProjectionView("left", np.eye(3), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="intrinsic"):
        ProjectionView("front", np.zeros((3, 3)), np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="rotation"):
        ProjectionView("front", np.eye(3), np.ones((3, 3)), np.zeros(3))
    with pytest.raises(ValueError, match="translation"):
        ProjectionView("front", np.eye(3), np.eye(3), np.zeros(2))
    behind = tuple(
        ProjectionView(
            view.name,
            view.K,
            view.R_model_to_camera,
            np.array([0.0, 0.0, -10.0]),
        )
        for view in _views()
    )
    with pytest.raises(ValueError, match="positive depth.*front"):
        _build(
            shape_basis,
            vertices=vertices,
            support=support,
            protected=protected,
            views=behind,
        )
    with pytest.raises(ValueError, match="min_nasal_response_ratio"):
        ObservableFlameSubspaceConfig(min_nasal_response_ratio=1.1)
    with pytest.raises(ValueError, match="max_rank"):
        ObservableFlameSubspaceConfig(max_rank=0)


@pytest.mark.parametrize(
    "invalid_config",
    [False, 0, {}],
    ids=["false", "zero", "empty-mapping"],
)
def test_falsey_invalid_config_values_reach_type_validation(invalid_config):
    vertices, shape_basis, support, protected = _problem(mode_count=1)
    shape_basis[:3, 0, 0] = 1.0

    with pytest.raises(
        ValueError,
        match="config must be an ObservableFlameSubspaceConfig",
    ):
        _build(
            shape_basis,
            config=invalid_config,
            vertices=vertices,
            support=support,
            protected=protected,
        )


def test_build_does_not_mutate_inputs_and_result_arrays_are_deeply_immutable():
    vertices, shape_basis, support, protected = _problem(mode_count=3)
    shape_basis[:3, 0, 0] = 1.0
    shape_basis[:3, 1, 1] = [0.2, 0.8, -0.1]
    views = _views()
    arrays = [vertices, shape_basis, support, protected]
    arrays.extend(
        value
        for view in views
        for value in (view.K, view.R_model_to_camera, view.t_model_to_camera)
    )
    snapshots = [value.copy() for value in arrays]

    result = _build(
        shape_basis,
        vertices=vertices,
        support=support,
        protected=protected,
        views=views,
    )

    for value, snapshot in zip(arrays, snapshots):
        np.testing.assert_array_equal(value, snapshot)
    public_arrays = (
        result.coefficient_basis,
        result.vertex_basis,
        result.candidate_mode_indices,
        result.screening_pass_mask,
        result.nasal_mean_squared_energy,
        result.outside_mean_squared_energy,
        result.protected_mean_squared_energy,
        result.nasal_response_ratio,
        result.outside_to_nasal_energy_ratio,
        result.protected_to_nasal_energy_ratio,
        result.relative_nasal_energy,
        result.singular_values,
    )
    for value in public_arrays:
        assert not value.flags.writeable
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            value.setflags(write=True)
    with pytest.raises(FrozenInstanceError):
        result.retained_rank = 99
    with pytest.raises(TypeError):
        result.report_data["status"] = "changed"
    json.dumps(result.to_report_data())


def test_projection_view_snapshots_sources_and_is_deeply_immutable():
    K = np.eye(3)
    K[0, 0] = 100.0
    K[1, 1] = 110.0
    rotation = np.eye(3)
    translation = np.array([0.1, -0.2, 3.0])
    expected = (K.copy(), rotation.copy(), translation.copy())
    view = ProjectionView("front", K, rotation, translation)
    K[:] = 9.0
    rotation[:] = 9.0
    translation[:] = 9.0

    for stored, snapshot in zip(
        (view.K, view.R_model_to_camera, view.t_model_to_camera),
        expected,
    ):
        np.testing.assert_array_equal(stored, snapshot)
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            stored.setflags(write=True)

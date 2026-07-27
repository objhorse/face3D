from __future__ import annotations

from dataclasses import FrozenInstanceError, replace

import numpy as np
import pytest
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from src.geometry.observable_flame_subspace import (
    ProjectionView,
    build_observable_flame_subspace,
)
from src.geometry.nasal_semantic_basis import (
    NASAL_SEMANTIC_MODE_NAMES,
    NasalSemanticBasisConfig,
    NasalSemanticFrame,
    apply_nasal_semantic_basis,
    build_nasal_semantic_basis,
)
from src.geometry.semantic_regions import build_default_nose_mouth_control_seeds


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _landmark_xy() -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.float64)
    jaw_x = np.linspace(-0.92, 0.92, 17)
    points[:17, 0] = jaw_x
    points[:17, 1] = -0.72 - 0.22 * (1.0 - (jaw_x / 0.92) ** 2)
    points[17:22] = np.column_stack(
        (np.linspace(-0.72, -0.18, 5), [0.48, 0.56, 0.58, 0.56, 0.50])
    )
    points[22:27] = np.column_stack(
        (np.linspace(0.18, 0.72, 5), [0.50, 0.56, 0.58, 0.56, 0.48])
    )
    points[27:31] = np.column_stack((np.zeros(4), [0.42, 0.27, 0.12, -0.03]))
    points[31:36] = np.array(
        [
            [-0.28, -0.11],
            [-0.15, -0.17],
            [0.00, -0.20],
            [0.15, -0.17],
            [0.28, -0.11],
        ]
    )
    eye = np.array(
        [
            [-0.21, 0.00],
            [-0.11, 0.06],
            [0.03, 0.06],
            [0.13, 0.00],
            [0.03, -0.05],
            [-0.11, -0.05],
        ]
    )
    points[36:42] = eye + np.array([-0.34, 0.28])
    points[42:48] = eye * np.array([-1.0, 1.0]) + np.array([0.34, 0.28])
    outer_angles = np.linspace(np.pi, -np.pi, 12, endpoint=False)
    points[48:60] = np.column_stack(
        (0.40 * np.cos(outer_angles), -0.48 + 0.15 * np.sin(outer_angles))
    )
    inner_angles = np.linspace(np.pi, -np.pi, 8, endpoint=False)
    points[60:68] = np.column_stack(
        (0.24 * np.cos(inner_angles), -0.48 + 0.07 * np.sin(inner_angles))
    )
    return points


def _synthetic_face():
    xs = np.linspace(-1.0, 1.0, 41)
    ys = np.linspace(-1.05, 0.95, 41)
    xx, yy = np.meshgrid(xs, ys)
    zz = (
        0.08
        + 0.24 * np.exp(-((xx / 0.32) ** 2 + ((yy + 0.02) / 0.42) ** 2))
        + 0.07 * np.exp(-((xx / 0.13) ** 2 + ((yy + 0.10) / 0.16) ** 2))
    )
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel())).astype(np.float64)
    faces = []
    width = len(xs)
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            a = row * width + col
            b = a + 1
            c = a + width
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    faces = np.asarray(faces, dtype=np.int64)

    triangles = np.empty((68, 3), dtype=np.int64)
    barycentric = np.zeros((68, 3), dtype=np.float64)
    barycentric[:, 0] = 1.0
    for index, point in enumerate(_landmark_xy()):
        vertex = int(np.argmin(np.linalg.norm(vertices[:, :2] - point, axis=1)))
        face = faces[np.flatnonzero(np.any(faces == vertex, axis=1))[0]]
        triangles[index] = np.concatenate(([vertex], face[face != vertex]))
    return vertices, faces, triangles, barycentric


def _build_basis(vertices, faces, triangles, barycentric, **kwargs):
    return build_nasal_semantic_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        model_to_front_camera=kwargs.pop(
            "model_to_front_camera", np.eye(3, dtype=np.float64)
        ),
        **kwargs,
    )


@pytest.fixture
def synthetic_basis():
    vertices, faces, triangles, barycentric = _synthetic_face()
    basis = _build_basis(vertices, faces, triangles, barycentric)
    return vertices, faces, triangles, barycentric, basis


def _mode_index(basis, name: str) -> int:
    return basis.names.index(name)


def _weighted_projection(basis, mode: str, direction: np.ndarray, mask: np.ndarray) -> float:
    index = _mode_index(basis, mode)
    weights = basis.weights[index] * mask
    assert float(weights.sum()) > 0.0
    projection = basis.vectors[index] @ direction
    return float(np.sum(weights * projection) / np.sum(weights))


def test_basis_names_dimensions_finite_deterministic_and_immutable():
    vertices, faces, triangles, barycentric = _synthetic_face()
    first = _build_basis(vertices, faces, triangles, barycentric)
    second = _build_basis(vertices, faces, triangles, barycentric)

    assert first.names == NASAL_SEMANTIC_MODE_NAMES == (
        "alar_width_shared",
        "alar_width_asymmetry",
        "alar_depth_shared",
        "alar_depth_asymmetry",
        "tip_depth",
        "tip_vertical",
        "tip_roundness",
        "tip_alar_fullness",
    )
    assert first.vectors.shape == (8, len(vertices), 3)
    assert first.weights.shape == (8, len(vertices))
    assert first.directions.shape == (8, len(vertices), 3)
    assert first.mode_support_masks.shape == (8, len(vertices))
    assert first.orthogonalization_weights.shape == (len(vertices),)
    assert first.scales.shape == (8,)
    assert np.isfinite(first.vectors).all()
    assert np.isfinite(first.weights).all()
    assert np.isfinite(first.directions).all()
    np.testing.assert_array_equal(first.vectors, second.vectors)
    np.testing.assert_array_equal(first.weights, second.weights)
    np.testing.assert_array_equal(first.semantic_frame.matrix, second.semantic_frame.matrix)
    active_direction_norms = np.linalg.norm(first.directions, axis=2)[first.weights > 0.0]
    np.testing.assert_allclose(active_direction_norms, 1.0, atol=1e-10)
    assert all(not value.flags.writeable for value in (
        first.vectors,
        first.weights,
        first.directions,
        first.mode_support_masks,
        first.protected_mask,
        first.support_mask,
        first.scales,
        first.orthogonalization_weights,
    ))
    with pytest.raises(FrozenInstanceError):
        first.face_width = 1.0
    with pytest.raises(ValueError):
        first.vectors[0, 0, 0] = 1.0
    with pytest.raises(TypeError):
        first.seed_indices["new"] = np.array([0])


def test_all_public_arrays_have_non_writeable_backing_storage(synthetic_basis):
    _vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    public_arrays = {
        "vectors": basis.vectors,
        "weights": basis.weights,
        "directions": basis.directions,
        "scales": basis.scales,
        "mode_support_masks": basis.mode_support_masks,
        "protected_mask": basis.protected_mask,
        "support_mask": basis.support_mask,
        "orthogonalization_weights": basis.orthogonalization_weights,
        "frame_origin": basis.semantic_frame.origin,
        "frame_matrix": basis.semantic_frame.matrix,
        "frame_rotation": basis.semantic_frame.model_to_front_camera,
        "frame_subject_left": basis.semantic_frame.subject_left,
        "frame_up": basis.semantic_frame.up,
        "frame_depth": basis.semantic_frame.depth,
    }
    public_arrays.update(
        {f"region:{name}": value for name, value in basis.region_masks.items()}
    )
    public_arrays.update(
        {f"seed:{name}": value for name, value in basis.seed_indices.items()}
    )

    for name, value in public_arrays.items():
        assert not value.flags.writeable, name
        with pytest.raises(ValueError, match="WRITEABLE|writeable"):
            value.setflags(write=True)


def test_caller_owned_array_mutation_cannot_affect_stored_basis(synthetic_basis):
    _vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    array_sources = {
        "vectors": basis.vectors.copy(),
        "weights": basis.weights.copy(),
        "directions": basis.directions.copy(),
        "scales": basis.scales.copy(),
        "mode_support_masks": basis.mode_support_masks.copy(),
        "protected_mask": basis.protected_mask.copy(),
        "support_mask": basis.support_mask.copy(),
        "orthogonalization_weights": basis.orthogonalization_weights.copy(),
    }
    region_sources = {
        name: value.copy() for name, value in basis.region_masks.items()
    }
    seed_sources = {
        name: value.copy() for name, value in basis.seed_indices.items()
    }
    origin_source = basis.semantic_frame.origin.copy()
    matrix_source = basis.semantic_frame.matrix.copy()
    rotation_source = basis.semantic_frame.model_to_front_camera.copy()
    frame = NasalSemanticFrame(
        origin_source,
        matrix_source,
        rotation_source,
    )
    expected_arrays = {name: value.copy() for name, value in array_sources.items()}
    expected_regions = {
        name: value.copy() for name, value in region_sources.items()
    }
    expected_seeds = {
        name: value.copy() for name, value in seed_sources.items()
    }
    expected_frame = (
        origin_source.copy(),
        matrix_source.copy(),
        rotation_source.copy(),
    )
    frozen = replace(
        basis,
        semantic_frame=frame,
        region_masks=region_sources,
        seed_indices=seed_sources,
        **array_sources,
    )

    for value in array_sources.values():
        value.flat[0] = not value.flat[0] if value.dtype == bool else value.flat[0] + 1
    for value in region_sources.values():
        value.flat[0] = not value.flat[0]
    for value in seed_sources.values():
        value.flat[0] = 0
    region_sources.clear()
    seed_sources.clear()
    origin_source[:] = 100.0
    matrix_source[:] = 100.0
    rotation_source[:] = 100.0

    for name, expected in expected_arrays.items():
        np.testing.assert_array_equal(getattr(frozen, name), expected)
    for name, expected in expected_regions.items():
        np.testing.assert_array_equal(frozen.region_masks[name], expected)
    for name, expected in expected_seeds.items():
        np.testing.assert_array_equal(frozen.seed_indices[name], expected)
    np.testing.assert_array_equal(frozen.semantic_frame.origin, expected_frame[0])
    np.testing.assert_array_equal(frozen.semantic_frame.matrix, expected_frame[1])
    np.testing.assert_array_equal(
        frozen.semantic_frame.model_to_front_camera,
        expected_frame[2],
    )


def test_support_is_a_compact_physical_geodesic_neighborhood(synthetic_basis):
    vertices, faces, _triangles, _barycentric, basis = synthetic_basis
    edges = _mesh_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 0]] - vertices[edges[:, 1]], axis=1)
    graph = coo_matrix(
        (
            np.concatenate((lengths, lengths)),
            (
                np.concatenate((edges[:, 0], edges[:, 1])),
                np.concatenate((edges[:, 1], edges[:, 0])),
            ),
        ),
        shape=(len(vertices), len(vertices)),
    ).tocsr()
    nasal_seeds = np.unique(
        np.concatenate(
            [
                basis.seed_indices["nose_bridge"],
                basis.seed_indices["nose_tip"],
                basis.seed_indices["subject_left_nose_wing"],
                basis.seed_indices["subject_right_nose_wing"],
            ]
        )
    )
    distance = dijkstra(graph, indices=nasal_seeds, directed=False, min_only=True)
    maximum_radius = basis.face_width * basis.config.support_radius_ratio

    assert np.any(basis.support_mask)
    assert np.all(distance[basis.support_mask] <= maximum_radius + 1e-12)
    assert np.all(~basis.mode_support_masks[:, ~basis.support_mask])
    assert np.count_nonzero(basis.support_mask) < len(vertices) // 4


def test_protected_landmark_triangle_vertices_have_exactly_zero_motion(synthetic_basis):
    _vertices, _faces, triangles, _barycentric, basis = synthetic_basis
    protected_landmarks = np.r_[0:17, 33, 36:48, 48:68]
    protected_vertices = np.unique(triangles[protected_landmarks].reshape(-1))
    helper_seeds = build_default_nose_mouth_control_seeds(triangles)

    assert np.all(basis.protected_mask[protected_vertices])
    assert np.all(basis.vectors[:, protected_vertices, :] == 0.0)
    assert np.all(basis.weights[:, protected_vertices] == 0.0)
    for name in (
        "philtrum",
        "subject_right_mouth_corner",
        "upper_lip",
        "subject_left_mouth_corner",
        "lower_lip",
    ):
        assert np.all(basis.protected_mask[helper_seeds[name]])


def test_zero_coefficients_return_an_exact_baseline_copy(synthetic_basis):
    vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    baseline = vertices.astype(np.float32)
    candidate, displacement = apply_nasal_semantic_basis(
        baseline, basis, np.zeros(8, dtype=np.float64)
    )

    assert candidate is not baseline
    np.testing.assert_array_equal(candidate, baseline)
    np.testing.assert_array_equal(displacement, np.zeros_like(baseline))
    assert candidate.dtype == baseline.dtype


def test_alar_shared_and_asymmetry_have_subject_relative_signs(synthetic_basis):
    vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    left_axis = basis.semantic_frame.subject_left
    depth_axis = basis.semantic_frame.depth
    left = basis.region_masks["subject_left_nose_wing"]
    right = basis.region_masks["subject_right_nose_wing"]

    assert _weighted_projection(basis, "alar_width_shared", left_axis, left) > 0.0
    assert _weighted_projection(basis, "alar_width_shared", -left_axis, right) > 0.0
    assert _weighted_projection(basis, "alar_width_asymmetry", left_axis, left) > 0.0
    assert _weighted_projection(basis, "alar_width_asymmetry", -left_axis, right) < 0.0
    assert _weighted_projection(basis, "alar_depth_shared", depth_axis, left) > 0.0
    assert _weighted_projection(basis, "alar_depth_shared", depth_axis, right) > 0.0
    assert _weighted_projection(basis, "alar_depth_asymmetry", depth_axis, left) > 0.0
    assert _weighted_projection(basis, "alar_depth_asymmetry", depth_axis, right) < 0.0
    for name in basis.names[:4]:
        coefficients = np.zeros(8)
        coefficients[_mode_index(basis, name)] = 0.4
        _positive_candidate, positive = apply_nasal_semantic_basis(
            vertices, basis, coefficients
        )
        _negative_candidate, negative = apply_nasal_semantic_basis(
            vertices, basis, -coefficients
        )
        np.testing.assert_allclose(negative, -positive, atol=1e-12)


def test_tip_mode_positive_and_negative_semantics(synthetic_basis):
    vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    tip = basis.region_masks["nose_tip"]
    transition = basis.region_masks["tip_alar_transition"]
    depth = basis.semantic_frame.depth
    up = basis.semantic_frame.up
    radial = vertices - basis.semantic_frame.origin
    radial -= (radial @ depth)[:, None] * depth
    radial_norm = np.linalg.norm(radial, axis=1)
    radial_direction = np.zeros_like(radial)
    valid = radial_norm > 1e-12
    radial_direction[valid] = radial[valid] / radial_norm[valid, None]

    semantic_scores = {
        "tip_depth": lambda delta: np.sum(
            basis.weights[_mode_index(basis, "tip_depth")] * (delta @ depth)
        ),
        "tip_vertical": lambda delta: np.sum(
            basis.weights[_mode_index(basis, "tip_vertical")] * (delta @ up)
        ),
        "tip_roundness": lambda delta: np.sum(
            basis.weights[_mode_index(basis, "tip_roundness")]
            * np.sum(delta * radial_direction, axis=1)
        ),
        "tip_alar_fullness": lambda delta: np.sum(
            transition * (delta @ depth)
        ),
    }
    assert np.any(tip)
    assert np.any(transition)
    for name, score in semantic_scores.items():
        coefficients = np.zeros(8)
        coefficients[_mode_index(basis, name)] = 0.4
        _positive_candidate, positive = apply_nasal_semantic_basis(
            vertices, basis, coefficients
        )
        _negative_candidate, negative = apply_nasal_semantic_basis(
            vertices, basis, -coefficients
        )
        assert score(positive) > 0.0, name
        assert score(negative) < 0.0, name
        np.testing.assert_allclose(negative, -positive, atol=1e-12)


def test_front_camera_rotation_controls_positive_depth_direction():
    vertices, faces, triangles, barycentric = _synthetic_face()
    angle = np.deg2rad(30.0)
    rotation = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    basis = build_nasal_semantic_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        model_to_front_camera=rotation,
    )
    expected_depth = rotation.T @ np.array([0.0, 0.0, -1.0])
    tip_depth = basis.vectors[_mode_index(basis, "tip_depth")]

    np.testing.assert_allclose(basis.semantic_frame.depth, expected_depth, atol=1e-12)
    assert np.sum(tip_depth @ expected_depth) > 0.0
    assert np.any(np.abs(tip_depth @ np.array([1.0, 0.0, 0.0])) > 0.0)


def test_front_camera_rotation_is_required_and_near_rotation_is_projected():
    vertices, faces, triangles, barycentric = _synthetic_face()
    with pytest.raises(TypeError, match="model_to_front_camera"):
        build_nasal_semantic_basis(vertices, faces, triangles, barycentric)

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
        target_determinant, abs=1e-12
    )
    assert error == pytest.approx(4.41e-7, abs=1e-12)

    basis = _build_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        model_to_front_camera=real_like,
    )
    projected = basis.semantic_frame.model_to_front_camera
    np.testing.assert_allclose(projected @ projected.T, np.eye(3), atol=1e-12)
    assert np.linalg.det(projected) == pytest.approx(1.0, abs=1e-12)
    assert np.linalg.norm(projected - real_like) < 1e-5

    reflection = np.diag([-1.0, 1.0, 1.0])
    with pytest.raises(ValueError, match="determinant|reflection"):
        _build_basis(
            vertices,
            faces,
            triangles,
            barycentric,
            model_to_front_camera=reflection,
        )


def test_semantic_masks_build_a_valid_observable_flame_subspace():
    vertices, faces, triangles, barycentric = _synthetic_face()
    angle = np.deg2rad(12.0)
    exact_front = np.array(
        [
            [np.cos(angle), 0.0, np.sin(angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(angle), 0.0, np.cos(angle)],
        ]
    )
    target_determinant = 0.99999968
    first_scale = np.sqrt(1.0 - 4.41e-7)
    real_like_front = (
        np.diag([first_scale, target_determinant / first_scale, 1.0])
        @ exact_front
    )
    semantic_basis = _build_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        model_to_front_camera=real_like_front,
    )
    intrinsic = np.array(
        [
            [820.0, 24.0, 320.0],
            [0.0, 910.0, 240.0],
            [0.0, 0.0, 1.0],
        ]
    )
    front_view = ProjectionView(
        "front",
        intrinsic,
        real_like_front,
        np.array([0.0, 0.0, 4.0]),
    )
    side_angle = np.deg2rad(27.0)
    side_left = np.array(
        [
            [np.cos(side_angle), 0.0, np.sin(side_angle)],
            [0.0, 1.0, 0.0],
            [-np.sin(side_angle), 0.0, np.cos(side_angle)],
        ]
    )
    views = (
        ProjectionView(
            "subject-right",
            intrinsic,
            side_left.T @ front_view.R_model_to_camera,
            np.array([0.0, 0.0, 4.0]),
        ),
        front_view,
        ProjectionView(
            "subject-left",
            intrinsic,
            side_left @ front_view.R_model_to_camera,
            np.array([0.0, 0.0, 4.0]),
        ),
    )
    flame_shape_basis = np.transpose(semantic_basis.vectors, (1, 2, 0))

    result = build_observable_flame_subspace(
        vertices,
        flame_shape_basis,
        semantic_basis.support_mask,
        semantic_basis.protected_mask,
        views,
    )

    np.testing.assert_allclose(
        front_view.R_model_to_camera,
        semantic_basis.semantic_frame.model_to_front_camera,
        atol=1e-14,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        front_view.R_model_to_camera @ front_view.R_model_to_camera.T,
        np.eye(3),
        atol=1e-12,
    )
    assert np.linalg.det(front_view.R_model_to_camera) == pytest.approx(1.0)
    assert result.candidate_mode_indices.size > 0
    assert result.retained_rank > 0
    np.testing.assert_allclose(
        result.coefficient_basis.T @ result.coefficient_basis,
        np.eye(result.retained_rank),
        atol=1e-12,
    )
    assert result.vertex_basis.shape == (
        len(vertices),
        3,
        result.retained_rank,
    )
    assert result.report_data["status"] == "ok"


def test_all_modes_have_low_pairwise_support_weighted_cosine(synthetic_basis):
    _vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    metric = basis.orthogonalization_weights
    gram = np.einsum(
        "v,ivc,jvc->ij",
        metric,
        basis.vectors,
        basis.vectors,
    )
    norms = np.sqrt(np.diag(gram))
    cosine = gram / np.outer(norms, norms)
    off_diagonal = np.abs(cosine - np.eye(len(basis.names)))
    threshold = basis.config.max_pairwise_weighted_cosine

    assert float(np.max(off_diagonal)) <= threshold
    fullness = _mode_index(basis, "tip_alar_fullness")
    tip_depth = _mode_index(basis, "tip_depth")
    alar_depth = _mode_index(basis, "alar_depth_shared")
    assert abs(float(cosine[fullness, tip_depth])) <= threshold
    assert abs(float(cosine[fullness, alar_depth])) <= threshold


def test_roundness_is_not_collinear_with_tip_depth(synthetic_basis):
    _vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    depth = basis.vectors[_mode_index(basis, "tip_depth")].reshape(-1)
    roundness = basis.vectors[_mode_index(basis, "tip_roundness")].reshape(-1)
    cosine = abs(float(depth @ roundness)) / (
        float(np.linalg.norm(depth) * np.linalg.norm(roundness))
    )

    assert np.linalg.norm(roundness) > 0.0
    assert cosine < 0.35


def test_tip_alar_fullness_stays_in_transition_not_mouth_or_cheek(synthetic_basis):
    vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    index = _mode_index(basis, "tip_alar_fullness")
    active = basis.mode_support_masks[index]
    transition = basis.region_masks["tip_alar_transition"]
    left_projection = (vertices - basis.semantic_frame.origin) @ basis.semantic_frame.subject_left

    assert np.any(active)
    assert np.all(active <= transition)
    assert np.max(np.abs(left_projection[active])) < basis.face_width * 0.25
    assert np.all(basis.vectors[index, basis.protected_mask] == 0.0)


def test_unit_mode_amplitude_is_fixed_face_width_ratio(synthetic_basis):
    _vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    expected = basis.face_width * basis.config.unit_displacement_ratio
    maxima = np.max(np.linalg.norm(basis.vectors, axis=2), axis=1)

    np.testing.assert_allclose(basis.scales, expected, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(maxima, expected, rtol=1e-10, atol=1e-12)


def test_small_combinations_are_linear_continuous_and_edge_smooth(synthetic_basis):
    vertices, faces, _triangles, _barycentric, basis = synthetic_basis
    coefficients = np.array([0.18, -0.12, 0.09, -0.07, 0.15, 0.08, 0.11, 0.13])
    candidate, displacement = apply_nasal_semantic_basis(vertices, basis, coefficients)
    half_candidate, half_displacement = apply_nasal_semantic_basis(
        vertices, basis, coefficients * 0.5
    )
    edges = _mesh_edges(faces)
    jumps = np.linalg.norm(
        displacement[edges[:, 0]] - displacement[edges[:, 1]], axis=1
    )
    neighbors = [[] for _ in range(len(vertices))]
    for a, b in edges:
        neighbors[int(a)].append(int(b))
        neighbors[int(b)].append(int(a))
    laplacian = np.zeros_like(displacement)
    for vertex, adjacent in enumerate(neighbors):
        if adjacent:
            laplacian[vertex] = displacement[vertex] - displacement[adjacent].mean(axis=0)

    np.testing.assert_allclose(candidate, vertices + displacement, atol=1e-12)
    np.testing.assert_allclose(half_displacement, displacement * 0.5, atol=1e-12)
    np.testing.assert_allclose(half_candidate, vertices + displacement * 0.5, atol=1e-12)
    assert float(np.max(jumps)) < basis.unit_scale * 0.55
    assert float(np.percentile(np.linalg.norm(laplacian, axis=1), 99.0)) < basis.unit_scale * 0.25


def test_build_and_apply_never_mutate_inputs():
    vertices, faces, triangles, barycentric = _synthetic_face()
    rotation = np.eye(3, dtype=np.float64)
    build_inputs = (vertices, faces, triangles, barycentric, rotation)
    build_snapshots = tuple(value.copy() for value in build_inputs)
    basis = build_nasal_semantic_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        model_to_front_camera=rotation,
    )
    for value, snapshot in zip(build_inputs, build_snapshots):
        np.testing.assert_array_equal(value, snapshot)

    baseline = vertices.copy()
    coefficients = np.linspace(-0.2, 0.2, 8)
    baseline_snapshot = baseline.copy()
    coefficient_snapshot = coefficients.copy()
    _candidate, _displacement = apply_nasal_semantic_basis(
        baseline, basis, coefficients
    )

    np.testing.assert_array_equal(baseline, baseline_snapshot)
    np.testing.assert_array_equal(coefficients, coefficient_snapshot)


def test_basis_validation_rejects_invalid_region_and_seed_metadata(synthetic_basis):
    _vertices, _faces, _triangles, _barycentric, basis = synthetic_basis
    missing_region = dict(basis.region_masks)
    missing_region.pop("nose_bridge")
    with pytest.raises(ValueError, match="region mask keys"):
        replace(basis, region_masks=missing_region)

    bad_region_shape = dict(basis.region_masks)
    bad_region_shape["nose_bridge"] = np.zeros(2, dtype=bool)
    with pytest.raises(ValueError, match="region mask.*shape"):
        replace(basis, region_masks=bad_region_shape)

    protected_region = dict(basis.region_masks)
    protected_region["nose_bridge"] = protected_region["nose_bridge"].copy()
    protected_region["nose_bridge"][np.flatnonzero(basis.protected_mask)[0]] = True
    with pytest.raises(ValueError, match="protected"):
        replace(basis, region_masks=protected_region)

    missing_seed = dict(basis.seed_indices)
    missing_seed.pop("nose_tip")
    with pytest.raises(ValueError, match="seed index keys"):
        replace(basis, seed_indices=missing_seed)

    empty_seed = dict(basis.seed_indices)
    empty_seed["nose_tip"] = np.array([], dtype=np.int64)
    with pytest.raises(ValueError, match="non-empty integer 1-D"):
        replace(basis, seed_indices=empty_seed)

    floating_seed = dict(basis.seed_indices)
    floating_seed["nose_tip"] = np.array([1.0])
    with pytest.raises(ValueError, match="non-empty integer 1-D"):
        replace(basis, seed_indices=floating_seed)

    out_of_bounds_seed = dict(basis.seed_indices)
    out_of_bounds_seed["nose_tip"] = np.array([len(basis.support_mask)])
    with pytest.raises(ValueError, match="bounds"):
        replace(basis, seed_indices=out_of_bounds_seed)


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("vertices", np.zeros((5, 2)), "vertices"),
        ("faces", np.zeros((2, 4), dtype=np.int64), "faces"),
        ("triangles", np.zeros((67, 3), dtype=np.int64), "lmk_tri_vidx"),
        ("barycentric", np.zeros((68, 2)), "lmk_bary_coords"),
        ("rotation", np.ones((3, 3)), "rotation"),
    ],
)
def test_invalid_build_inputs_fail_clearly(field, value, match):
    vertices, faces, triangles, barycentric = _synthetic_face()
    arguments = {
        "vertices": vertices,
        "faces": faces,
        "lmk_tri_vidx": triangles,
        "lmk_bary_coords": barycentric,
        "model_to_front_camera": np.eye(3),
    }
    key = {
        "triangles": "lmk_tri_vidx",
        "barycentric": "lmk_bary_coords",
        "rotation": "model_to_front_camera",
    }.get(field, field)
    arguments[key] = value

    with pytest.raises(ValueError, match=match):
        build_nasal_semantic_basis(**arguments)


def test_invalid_indices_nonfinite_values_config_and_coefficients_fail(synthetic_basis):
    vertices, faces, triangles, barycentric, basis = synthetic_basis
    bad_vertices = vertices.copy()
    bad_vertices[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        _build_basis(bad_vertices, faces, triangles, barycentric)

    bad_faces = faces.copy()
    bad_faces[0, 0] = len(vertices)
    with pytest.raises(ValueError, match="indices"):
        _build_basis(vertices, bad_faces, triangles, barycentric)

    bad_barycentric = barycentric.copy()
    bad_barycentric[0] = [0.2, 0.2, 0.2]
    with pytest.raises(ValueError, match="sum"):
        _build_basis(vertices, faces, triangles, bad_barycentric)

    non_face_mapping = triangles.copy()
    non_face_mapping[30] = [0, len(vertices) // 2, len(vertices) - 1]
    with pytest.raises(ValueError, match="mesh faces"):
        _build_basis(
            vertices, faces, non_face_mapping, barycentric
        )

    with pytest.raises(ValueError, match="support_radius_ratio"):
        NasalSemanticBasisConfig(support_radius_ratio=0.0)
    with pytest.raises(ValueError, match="unit_displacement_ratio"):
        NasalSemanticBasisConfig(unit_displacement_ratio="invalid")
    with pytest.raises(FrozenInstanceError):
        basis.config.support_radius_ratio = 1.0

    with pytest.raises(ValueError, match="coefficient"):
        apply_nasal_semantic_basis(vertices, basis, np.zeros(7))
    invalid_coefficients = np.zeros(8)
    invalid_coefficients[3] = np.inf
    with pytest.raises(ValueError, match="finite"):
        apply_nasal_semantic_basis(vertices, basis, invalid_coefficients)
    with pytest.raises(ValueError, match="vertices"):
        apply_nasal_semantic_basis(vertices[:-1], basis, np.zeros(8))


@pytest.mark.parametrize(
    "invalid_config",
    [False, 0, {}],
    ids=["false", "zero", "empty-mapping"],
)
def test_falsey_invalid_config_values_reach_type_validation(invalid_config):
    vertices, faces, triangles, barycentric = _synthetic_face()

    with pytest.raises(
        ValueError,
        match="config must be a NasalSemanticBasisConfig",
    ):
        _build_basis(
            vertices,
            faces,
            triangles,
            barycentric,
            config=invalid_config,
        )

from __future__ import annotations

import numpy as np

from src.geometry.nasal_base_semantic_basis import (
    NASAL_BASE_MODE_NAMES,
    apply_nasal_base_semantic_basis,
    build_nasal_base_semantic_basis,
)
from tests.test_nasal_semantic_basis import _synthetic_face


def _basis():
    vertices, faces, triangles, barycentric = _synthetic_face()
    basis = build_nasal_base_semantic_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        np.eye(3),
    )
    return vertices, faces, triangles, barycentric, basis


def _mode_index(name: str) -> int:
    return NASAL_BASE_MODE_NAMES.index(name)


def _landmark_point(
    vertices: np.ndarray,
    triangles: np.ndarray,
    barycentric: np.ndarray,
    index: int,
) -> np.ndarray:
    return np.sum(
        vertices[triangles[index]]
        * barycentric[index, :, None],
        axis=0,
    )


def test_basis_is_eight_mode_compact_finite_and_deterministic():
    vertices, faces, triangles, barycentric, first = _basis()
    second = build_nasal_base_semantic_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        np.eye(3),
    )

    assert first.names == NASAL_BASE_MODE_NAMES
    assert first.vectors.shape == (8, len(vertices), 3)
    assert np.isfinite(first.vectors).all()
    assert np.array_equal(first.vectors, second.vectors)
    assert np.any(first.support_mask)
    assert np.any(~first.support_mask)
    assert np.all(first.vectors[:, ~first.support_mask, :] == 0.0)


def test_outer_alar_tip_and_non_nasal_landmarks_are_exactly_fixed():
    vertices, _faces, triangles, _barycentric, basis = _basis()
    protected_landmarks = (0, 30, 31, 35, 48, 54)
    protected_vertices = np.unique(
        triangles[np.asarray(protected_landmarks)].reshape(-1)
    )

    assert np.all(basis.protected_mask[protected_vertices])
    assert np.all(basis.vectors[:, protected_vertices, :] == 0.0)


def test_shared_nostril_width_moves_inner_rims_apart_subject_relatively():
    vertices, _faces, triangles, barycentric, basis = _basis()
    coefficients = np.zeros(8)
    coefficients[_mode_index("nostril_width_shared")] = 1.0
    candidate = apply_nasal_base_semantic_basis(
        vertices,
        basis,
        coefficients,
    )
    before_right = _landmark_point(vertices, triangles, barycentric, 32)
    before_left = _landmark_point(vertices, triangles, barycentric, 34)
    after_right = _landmark_point(candidate, triangles, barycentric, 32)
    after_left = _landmark_point(candidate, triangles, barycentric, 34)
    axis = basis.semantic_frame.subject_left

    assert np.dot(after_left - before_left, axis) > 0.0
    assert np.dot(after_right - before_right, axis) < 0.0


def test_zero_coefficients_return_exact_input_and_inputs_are_not_mutated():
    vertices, _faces, _triangles, _barycentric, basis = _basis()
    original = vertices.copy()

    candidate = apply_nasal_base_semantic_basis(
        vertices,
        basis,
        np.zeros(8),
    )

    np.testing.assert_array_equal(candidate, original)
    np.testing.assert_array_equal(vertices, original)

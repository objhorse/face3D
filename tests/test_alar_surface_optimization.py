from __future__ import annotations

import numpy as np

from src.geometry.alar_surface_basis import (
    ALAR_SURFACE_MODE_NAMES,
    apply_alar_surface_basis,
    build_alar_surface_basis,
)
from src.geometry.alar_surface_observations import (
    _profile_alar_segment,
    _rasterized_distance,
)


def _landmark_xy() -> np.ndarray:
    points = np.zeros((68, 2), dtype=np.float64)
    jaw_x = np.linspace(-0.92, 0.92, 17)
    points[:17] = np.column_stack(
        (jaw_x, -0.72 - 0.22 * (1.0 - (jaw_x / 0.92) ** 2))
    )
    points[17:22] = np.column_stack(
        (np.linspace(-0.72, -0.18, 5), [0.48, 0.56, 0.58, 0.56, 0.50])
    )
    points[22:27] = np.column_stack(
        (np.linspace(0.18, 0.72, 5), [0.50, 0.56, 0.58, 0.56, 0.48])
    )
    points[27:31] = np.column_stack((np.zeros(4), [0.42, 0.27, 0.12, -0.03]))
    points[31:36] = np.asarray(
        [
            [-0.28, -0.11],
            [-0.15, -0.17],
            [0.00, -0.20],
            [0.15, -0.17],
            [0.28, -0.11],
        ]
    )
    eye = np.asarray(
        [
            [-0.21, 0.00],
            [-0.11, 0.06],
            [0.03, 0.06],
            [0.13, 0.00],
            [0.03, -0.05],
            [-0.11, -0.05],
        ]
    )
    points[36:42] = eye + np.asarray([-0.34, 0.28])
    points[42:48] = eye * np.asarray([-1.0, 1.0]) + np.asarray([0.34, 0.28])
    outer = np.linspace(np.pi, -np.pi, 12, endpoint=False)
    points[48:60] = np.column_stack(
        (0.40 * np.cos(outer), -0.48 + 0.15 * np.sin(outer))
    )
    inner = np.linspace(np.pi, -np.pi, 8, endpoint=False)
    points[60:68] = np.column_stack(
        (0.24 * np.cos(inner), -0.48 + 0.07 * np.sin(inner))
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
    vertices = np.column_stack((xx.ravel(), yy.ravel(), zz.ravel()))
    faces = []
    width = len(xs)
    for row in range(len(ys) - 1):
        for column in range(len(xs) - 1):
            a = row * width + column
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
        triangles[index] = np.r_[vertex, face[face != vertex]]
    return vertices, faces, triangles, barycentric


def test_signed_distance_preserves_outer_side():
    curve = np.column_stack(
        (np.full(20, 16.0), np.linspace(5.0, 24.0, 20))
    )
    field = _rasterized_distance(curve, (30, 32), outward_sign=1.0)
    assert field[15, 22] > 0.0
    assert field[15, 10] < 0.0
    assert abs(float(field[15, 16])) < 1e-8


def test_profile_segment_ends_at_alar_anchor():
    curve = np.column_stack(
        (
            20.0 + np.sin(np.linspace(0.0, np.pi, 60)) * 8.0,
            np.linspace(5.0, 55.0, 60),
        )
    )
    anchor = curve[-1]
    segment = _profile_alar_segment(curve, anchor, 0.4)
    np.testing.assert_allclose(segment[-1], anchor)
    assert 4 <= len(segment) < len(curve)


def test_alar_basis_moves_outer_wings_but_preserves_other_vertices():
    vertices, faces, triangles, barycentric = _synthetic_face()
    basis = build_alar_surface_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        np.eye(3),
    )
    assert basis.names == ALAR_SURFACE_MODE_NAMES
    assert basis.vectors.shape == (6, len(vertices), 3)
    assert np.count_nonzero(basis.support_mask) > 0
    candidate = apply_alar_surface_basis(
        vertices,
        basis,
        np.asarray([1.0, 0.25, 0.4, -0.2, 0.1, 0.35]),
    )
    assert np.array_equal(
        candidate[~basis.support_mask],
        vertices[~basis.support_mask],
    )
    assert np.array_equal(
        candidate[basis.protected_mask],
        vertices[basis.protected_mask],
    )
    assert np.any(candidate[basis.support_mask] != vertices[basis.support_mask])


def test_shared_width_moves_both_wings_outward_and_not_by_translation():
    vertices, faces, triangles, barycentric = _synthetic_face()
    basis = build_alar_surface_basis(
        vertices,
        faces,
        triangles,
        barycentric,
        np.eye(3),
    )
    displacement = basis.vectors[0]
    lateral = basis.semantic_frame.subject_left
    left = basis.region_masks["subject_left_alar"]
    right = basis.region_masks["subject_right_alar"]
    assert float(np.mean(displacement[left] @ lateral)) > 0.0
    assert float(np.mean(displacement[right] @ lateral)) < 0.0
    centroid_shift = np.mean(displacement[basis.support_mask], axis=0)
    assert np.linalg.norm(centroid_shift) < 0.25 * basis.unit_scale

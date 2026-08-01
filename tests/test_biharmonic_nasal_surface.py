from __future__ import annotations

import numpy as np

from src.geometry.biharmonic_nasal_surface import (
    BiharmonicNasalConfig,
    expand_vertex_mask,
    fit_biharmonic_nasal_surface,
    fit_vector_biharmonic_nasal_surface,
    fixed_inner_boundary_mask,
)


def _grid_mesh(size: int = 9) -> tuple[np.ndarray, np.ndarray]:
    vertices = np.asarray(
        [(float(x), float(y), 0.0) for y in range(size) for x in range(size)],
        dtype=np.float64,
    )
    faces = []
    for y in range(size - 1):
        for x in range(size - 1):
            first = y * size + x
            faces.append((first, first + 1, first + size + 1))
            faces.append((first, first + size + 1, first + size))
    return vertices, np.asarray(faces, dtype=np.int64)


def test_topology_mask_expansion_and_fixed_boundary_rings() -> None:
    vertices, faces = _grid_mesh()
    seed = np.zeros(len(vertices), dtype=bool)
    seed[4 * 9 + 4] = True
    expanded = expand_vertex_mask(seed, faces, rings=2)
    assert np.count_nonzero(expanded) > 1
    assert expanded[4 * 9 + 4]

    support = np.zeros(len(vertices), dtype=bool)
    for y in range(1, 8):
        for x in range(1, 8):
            support[y * 9 + x] = True
    fixed = fixed_inner_boundary_mask(support, faces, rings=2)
    assert fixed[1 * 9 + 1]
    assert fixed[2 * 9 + 2]
    assert not fixed[4 * 9 + 4]


def test_biharmonic_fit_reduces_error_without_boundary_discontinuity() -> None:
    vertices, faces = _grid_mesh()
    support = np.zeros(len(vertices), dtype=bool)
    for y in range(1, 8):
        for x in range(1, 8):
            support[y * 9 + x] = True
    selected_faces = []
    for center in ((3.5, 3.5), (4.5, 3.5), (3.5, 4.5), (4.5, 4.5)):
        centroids = vertices[faces].mean(axis=1)[:, :2]
        selected_faces.append(int(np.argmin(np.linalg.norm(centroids - center, axis=1))))
    selected_faces = np.asarray(selected_faces, dtype=np.int64)
    barycentric = np.full((4, 3), 1.0 / 3.0, dtype=np.float64)
    baseline_points = np.sum(
        vertices[faces[selected_faces]] * barycentric[:, :, None],
        axis=1,
    )
    target = baseline_points.copy()
    target[:, 2] = np.asarray([0.0, 0.02, 0.02, 0.0])

    result = fit_biharmonic_nasal_surface(
        vertices,
        faces,
        support,
        selected_faces,
        barycentric,
        target,
        np.ones(4, dtype=np.float64),
        config=BiharmonicNasalConfig(
            fixed_boundary_rings=2,
            bending_weight=2.0,
            displacement_prior_weight=0.01,
        ),
    )

    assert result.final_rmse_mm < result.initial_rmse_mm
    assert np.array_equal(result.candidate_vertices[~support], vertices[~support])
    assert np.array_equal(
        result.candidate_vertices[result.fixed_mask],
        vertices[result.fixed_mask],
    )
    scalar = result.scalar_displacement_m
    assert np.max(scalar) > 0.0
    edge_differences = []
    for first, second, third in faces:
        edge_differences.extend(
            (
                abs(scalar[first] - scalar[second]),
                abs(scalar[second] - scalar[third]),
                abs(scalar[third] - scalar[first]),
            )
        )
    assert max(edge_differences) < 0.01


def test_robust_fit_downweights_a_sparse_depth_outlier() -> None:
    vertices, faces = _grid_mesh()
    support = np.zeros(len(vertices), dtype=bool)
    for y in range(1, 8):
        for x in range(1, 8):
            support[y * 9 + x] = True
    centroids = vertices[faces].mean(axis=1)[:, :2]
    centers = ((3.2, 3.2), (4.8, 3.2), (3.2, 4.8), (4.8, 4.8), (4.0, 4.0))
    selected_faces = np.asarray(
        [int(np.argmin(np.linalg.norm(centroids - center, axis=1))) for center in centers],
        dtype=np.int64,
    )
    barycentric = np.full((len(selected_faces), 3), 1.0 / 3.0, dtype=np.float64)
    baseline_points = np.sum(
        vertices[faces[selected_faces]] * barycentric[:, :, None], axis=1
    )
    target = baseline_points.copy()
    target[:-1, 2] += 0.006
    target[-1, 2] += 0.060

    common = dict(
        baseline_vertices=vertices,
        faces=faces,
        support_mask=support,
        attachment_face_indices=selected_faces,
        attachment_barycentric=barycentric,
        target_points_model=target,
        observation_weights=np.ones(len(selected_faces), dtype=np.float64),
    )
    nonrobust = fit_biharmonic_nasal_surface(
        **common,
        config=BiharmonicNasalConfig(
            bending_weight=1.0,
            displacement_prior_weight=0.01,
            robust_delta_mm=1000.0,
        ),
    )
    robust = fit_biharmonic_nasal_surface(
        **common,
        config=BiharmonicNasalConfig(
            bending_weight=1.0,
            displacement_prior_weight=0.01,
            robust_delta_mm=2.0,
        ),
    )

    assert robust.metadata["downweighted_observation_count"] >= 1
    assert robust.metadata["maximum_displacement_mm"] < nonrobust.metadata[
        "maximum_displacement_mm"
    ]


def test_vector_fit_preserves_tangential_reference_without_moving_boundary() -> None:
    vertices, faces = _grid_mesh()
    support = np.zeros(len(vertices), dtype=bool)
    for y in range(1, 8):
        for x in range(1, 8):
            support[y * 9 + x] = True
    centroids = vertices[faces].mean(axis=1)[:, :2]
    centers = ((3.2, 3.2), (4.8, 3.2), (3.2, 4.8), (4.8, 4.8))
    selected_faces = np.asarray(
        [int(np.argmin(np.linalg.norm(centroids - center, axis=1))) for center in centers],
        dtype=np.int64,
    )
    barycentric = np.full((len(selected_faces), 3), 1.0 / 3.0, dtype=np.float64)
    baseline_points = np.sum(
        vertices[faces[selected_faces]] * barycentric[:, :, None], axis=1
    )
    reference = np.zeros_like(vertices)
    centered = vertices[:, :2] - np.asarray([4.0, 4.0])
    reference[support, 0] = centered[support, 0] * 0.002
    reference[support, 1] = centered[support, 1] * -0.001
    reference[support, 2] = (
        0.012 - 0.001 * np.sum(centered[support] ** 2, axis=1)
    )
    target = baseline_points + np.sum(
        reference[faces[selected_faces]] * barycentric[:, :, None], axis=1
    )

    result = fit_vector_biharmonic_nasal_surface(
        vertices,
        faces,
        support,
        selected_faces,
        barycentric,
        target,
        np.ones(len(selected_faces), dtype=np.float64),
        reference_displacement_m=reference,
        config=BiharmonicNasalConfig(
            bending_weight=0.01,
            displacement_prior_weight=0.01,
            robust_delta_mm=1000.0,
        ),
    )

    assert result.final_rmse_mm < result.initial_rmse_mm
    assert np.array_equal(result.candidate_vertices[~support], vertices[~support])
    assert np.array_equal(
        result.candidate_vertices[result.fixed_mask],
        vertices[result.fixed_mask],
    )
    assert np.max(np.abs(result.displacement_m[:, 0])) > 0.0
    assert np.max(np.abs(result.displacement_m[:, 1])) > 0.0
    assert result.metadata["solver"] == "sparse_normal_equations_3d"

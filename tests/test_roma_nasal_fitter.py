import numpy as np

from src.geometry.roma_nasal_fitter import RoMaNasalFitConfig, fit_roma_nasal_surface


def test_fitter_recovers_smooth_basis_coefficient() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    modes = np.zeros((1, 3, 3), dtype=np.float64)
    modes[0, :, 2] = np.asarray([0.0, 0.01, 0.02])
    count = 6
    face_indices = np.zeros(count, dtype=np.int64)
    barycentric = np.asarray(
        [
            [0.8, 0.1, 0.1],
            [0.6, 0.2, 0.2],
            [0.4, 0.3, 0.3],
            [0.2, 0.4, 0.4],
            [0.3, 0.6, 0.1],
            [0.3, 0.1, 0.6],
        ],
        dtype=np.float64,
    )
    baseline_points = np.sum(vertices[faces[face_indices]] * barycentric[:, :, None], axis=1)
    sampled_mode = np.sum(
        modes[0, faces[face_indices]] * barycentric[:, :, None],
        axis=1,
    )
    nuisance_translation = np.asarray([-0.002, 0.004, 0.012])
    targets = baseline_points + 1.5 * sampled_mode + nuisance_translation
    result = fit_roma_nasal_surface(
        vertices,
        faces,
        modes,
        face_indices,
        barycentric,
        targets,
        np.ones(count),
        config=RoMaNasalFitConfig(coefficient_prior_weight=1e-6),
    )
    assert result.success
    assert result.final_rmse_mm < result.initial_rmse_mm
    assert np.isclose(result.coefficients[0], 1.5, atol=0.1)
    assert np.allclose(
        result.metadata["nuisance_translation_mm"],
        nuisance_translation * 1000.0,
        atol=0.2,
    )


def test_fitter_keeps_vertices_outside_basis_exact() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5], [2.0, 2.0, 0.5]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    modes = np.zeros((1, 4, 3), dtype=np.float64)
    modes[0, :3, 2] = 0.01
    face_indices = np.zeros(6, dtype=np.int64)
    barycentric = np.tile(np.asarray([[0.4, 0.3, 0.3]]), (6, 1))
    baseline_points = np.sum(vertices[faces[face_indices]] * barycentric[:, :, None], axis=1)
    result = fit_roma_nasal_surface(
        vertices,
        faces,
        modes,
        face_indices,
        barycentric,
        baseline_points + np.asarray([0.0, 0.0, 0.01]),
        np.ones(6),
    )
    assert np.array_equal(result.candidate_vertices[3], vertices[3])


def test_fitter_orientation_barrier_prevents_target_driven_face_flip() -> None:
    vertices = np.asarray(
        [[0.0, 0.0, 0.5], [1.0, 0.0, 0.5], [0.0, 1.0, 0.5]],
        dtype=np.float64,
    )
    faces = np.asarray([[0, 1, 2]], dtype=np.int64)
    modes = np.zeros((1, 3, 3), dtype=np.float64)
    modes[0, 2, 1] = -2.0
    face_indices = np.zeros(6, dtype=np.int64)
    barycentric = np.asarray(
        [
            [0.05, 0.05, 0.90],
            [0.10, 0.10, 0.80],
            [0.15, 0.15, 0.70],
            [0.20, 0.20, 0.60],
            [0.25, 0.25, 0.50],
            [0.30, 0.30, 0.40],
        ],
        dtype=np.float64,
    )
    baseline_points = np.sum(
        vertices[faces[face_indices]] * barycentric[:, :, None],
        axis=1,
    )
    sampled_mode = np.sum(
        modes[0, faces[face_indices]] * barycentric[:, :, None],
        axis=1,
    )
    result = fit_roma_nasal_surface(
        vertices,
        faces,
        modes,
        face_indices,
        barycentric,
        baseline_points + 1.2 * sampled_mode,
        np.ones(6),
        config=RoMaNasalFitConfig(coefficient_prior_weight=1e-6),
    )
    baseline_cross = np.cross(vertices[1] - vertices[0], vertices[2] - vertices[0])
    candidate = result.candidate_vertices
    candidate_cross = np.cross(candidate[1] - candidate[0], candidate[2] - candidate[0])
    assert np.dot(baseline_cross, candidate_cross) > 0.0
    assert result.metadata["minimum_orientation_ratio"] > 0.0

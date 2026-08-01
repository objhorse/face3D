"""Smooth normal-displacement fitting for sparse multiview nasal evidence."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr, spsolve


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    topology = np.asarray(faces, dtype=np.int64)
    edges = np.vstack(
        (topology[:, [0, 1]], topology[:, [1, 2]], topology[:, [2, 0]])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _adjacency(vertex_count: int, faces: np.ndarray) -> tuple[np.ndarray, ...]:
    neighbors = [set() for _ in range(int(vertex_count))]
    for first, second in _unique_edges(faces):
        a = int(first)
        b = int(second)
        neighbors[a].add(b)
        neighbors[b].add(a)
    return tuple(np.asarray(sorted(values), dtype=np.int64) for values in neighbors)


def expand_vertex_mask(
    vertex_mask: np.ndarray,
    faces: np.ndarray,
    *,
    rings: int,
) -> np.ndarray:
    """Expand a semantic vertex region by a fixed number of topology rings."""
    mask = np.asarray(vertex_mask, dtype=bool).reshape(-1).copy()
    if isinstance(rings, bool) or int(rings) < 0:
        raise ValueError("rings must be a non-negative integer")
    edges = _unique_edges(faces)
    for _ in range(int(rings)):
        crossing = mask[edges[:, 0]] ^ mask[edges[:, 1]]
        if not np.any(crossing):
            break
        mask[np.unique(edges[crossing])] = True
    return mask


def fixed_inner_boundary_mask(
    support_mask: np.ndarray,
    faces: np.ndarray,
    *,
    rings: int,
) -> np.ndarray:
    """Return supported vertices in the requested number of outer rings."""
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    if isinstance(rings, bool) or int(rings) < 1:
        raise ValueError("rings must be a positive integer")
    adjacency = _adjacency(len(support), faces)
    fixed = np.zeros(len(support), dtype=bool)
    frontier = np.asarray(
        [
            index
            for index in np.flatnonzero(support)
            if np.any(~support[adjacency[int(index)]])
        ],
        dtype=np.int64,
    )
    for _ in range(int(rings)):
        if not len(frontier):
            break
        fixed[frontier] = True
        next_vertices = {
            int(neighbor)
            for vertex in frontier
            for neighbor in adjacency[int(vertex)]
            if support[int(neighbor)] and not fixed[int(neighbor)]
        }
        frontier = np.asarray(sorted(next_vertices), dtype=np.int64)
    return fixed


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices, dtype=np.float64)
    triangles = np.asarray(faces, dtype=np.int64)
    face_normals = np.cross(
        vertices[triangles[:, 1]] - vertices[triangles[:, 0]],
        vertices[triangles[:, 2]] - vertices[triangles[:, 0]],
    )
    for corner in range(3):
        np.add.at(normals, triangles[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    if np.any(lengths <= 1e-14):
        raise ValueError("baseline mesh contains vertices without usable normals")
    return normals / lengths


def project_normal_displacement(
    baseline_vertices: np.ndarray,
    reference_vertices: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    """Project a reference deformation onto baseline vertex normals."""
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    reference = np.asarray(reference_vertices, dtype=np.float64)
    if (
        baseline.ndim != 2
        or baseline.shape[1] != 3
        or reference.shape != baseline.shape
        or not np.isfinite(baseline).all()
        or not np.isfinite(reference).all()
    ):
        raise ValueError("baseline and reference vertices must have finite shape (V, 3)")
    normals = _vertex_normals(baseline, faces)
    return np.sum((reference - baseline) * normals, axis=1)


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    result = []
    for axis in range(values.shape[1]):
        order = np.argsort(values[:, axis])
        ordered_weights = weights[order]
        threshold = 0.5 * float(np.sum(ordered_weights))
        index = int(np.searchsorted(np.cumsum(ordered_weights), threshold))
        result.append(float(values[order[min(index, len(order) - 1)], axis]))
    return np.asarray(result, dtype=np.float64)


@dataclass(frozen=True)
class BiharmonicNasalConfig:
    fixed_boundary_rings: int = 2
    bending_weight: float = 12.0
    displacement_prior_weight: float = 0.2
    robust_delta_mm: float = 2.5
    robust_iterations: int = 5
    lsqr_tolerance: float = 1e-10
    max_iterations: int = 4000

    def __post_init__(self) -> None:
        if isinstance(self.fixed_boundary_rings, bool) or int(self.fixed_boundary_rings) < 1:
            raise ValueError("fixed_boundary_rings must be a positive integer")
        for name in (
            "bending_weight",
            "displacement_prior_weight",
            "robust_delta_mm",
            "lsqr_tolerance",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if isinstance(self.max_iterations, bool) or int(self.max_iterations) < 1:
            raise ValueError("max_iterations must be positive")
        if isinstance(self.robust_iterations, bool) or int(self.robust_iterations) < 1:
            raise ValueError("robust_iterations must be positive")


@dataclass(frozen=True)
class BiharmonicNasalResult:
    candidate_vertices: np.ndarray
    scalar_displacement_m: np.ndarray
    support_mask: np.ndarray
    fixed_mask: np.ndarray
    observation_mask: np.ndarray
    initial_rmse_mm: float
    final_rmse_mm: float
    nuisance_translation_mm: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.candidate_vertices, dtype=np.float64)
        displacement = np.asarray(self.scalar_displacement_m, dtype=np.float64)
        support = np.asarray(self.support_mask, dtype=bool)
        fixed = np.asarray(self.fixed_mask, dtype=bool)
        observations = np.asarray(self.observation_mask, dtype=bool)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("candidate_vertices must have finite shape (V, 3)")
        if displacement.shape != (len(vertices),) or not np.isfinite(displacement).all():
            raise ValueError("scalar_displacement_m must match candidate vertices")
        if support.shape != displacement.shape or fixed.shape != displacement.shape:
            raise ValueError("support and fixed masks must match candidate vertices")
        if observations.ndim != 1:
            raise ValueError("observation_mask must be one-dimensional")
        object.__setattr__(self, "candidate_vertices", vertices.copy())
        object.__setattr__(self, "scalar_displacement_m", displacement.copy())
        object.__setattr__(self, "support_mask", support.copy())
        object.__setattr__(self, "fixed_mask", fixed.copy())
        object.__setattr__(self, "observation_mask", observations.copy())
        object.__setattr__(
            self,
            "nuisance_translation_mm",
            np.asarray(self.nuisance_translation_mm, dtype=np.float64).copy(),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_rmse_mm": float(self.initial_rmse_mm),
            "final_rmse_mm": float(self.final_rmse_mm),
            "nuisance_translation_mm": self.nuisance_translation_mm.tolist(),
            "support_vertex_count": int(np.count_nonzero(self.support_mask)),
            "fixed_vertex_count": int(np.count_nonzero(self.fixed_mask)),
            "free_vertex_count": int(
                np.count_nonzero(self.support_mask & ~self.fixed_mask)
            ),
            "used_observation_count": int(np.count_nonzero(self.observation_mask)),
            "metadata": dict(self.metadata),
        }


@dataclass(frozen=True)
class VectorBiharmonicNasalResult:
    candidate_vertices: np.ndarray
    displacement_m: np.ndarray
    support_mask: np.ndarray
    fixed_mask: np.ndarray
    observation_mask: np.ndarray
    initial_rmse_mm: float
    final_rmse_mm: float
    nuisance_translation_mm: np.ndarray
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.candidate_vertices, dtype=np.float64)
        displacement = np.asarray(self.displacement_m, dtype=np.float64)
        support = np.asarray(self.support_mask, dtype=bool)
        fixed = np.asarray(self.fixed_mask, dtype=bool)
        observations = np.asarray(self.observation_mask, dtype=bool)
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("candidate_vertices must have finite shape (V, 3)")
        if displacement.shape != vertices.shape or not np.isfinite(displacement).all():
            raise ValueError("displacement_m must match candidate vertices")
        if support.shape != (len(vertices),) or fixed.shape != support.shape:
            raise ValueError("support and fixed masks must match candidate vertices")
        if observations.ndim != 1:
            raise ValueError("observation_mask must be one-dimensional")
        object.__setattr__(self, "candidate_vertices", vertices.copy())
        object.__setattr__(self, "displacement_m", displacement.copy())
        object.__setattr__(self, "support_mask", support.copy())
        object.__setattr__(self, "fixed_mask", fixed.copy())
        object.__setattr__(self, "observation_mask", observations.copy())
        object.__setattr__(
            self,
            "nuisance_translation_mm",
            np.asarray(self.nuisance_translation_mm, dtype=np.float64).copy(),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "initial_rmse_mm": float(self.initial_rmse_mm),
            "final_rmse_mm": float(self.final_rmse_mm),
            "nuisance_translation_mm": self.nuisance_translation_mm.tolist(),
            "support_vertex_count": int(np.count_nonzero(self.support_mask)),
            "fixed_vertex_count": int(np.count_nonzero(self.fixed_mask)),
            "free_vertex_count": int(
                np.count_nonzero(self.support_mask & ~self.fixed_mask)
            ),
            "used_observation_count": int(np.count_nonzero(self.observation_mask)),
            "metadata": dict(self.metadata),
        }


def fit_biharmonic_nasal_surface(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    support_mask: np.ndarray,
    attachment_face_indices: np.ndarray,
    attachment_barycentric: np.ndarray,
    target_points_model: np.ndarray,
    observation_weights: np.ndarray,
    *,
    reference_scalar_displacement_m: np.ndarray | None = None,
    config: BiharmonicNasalConfig | None = None,
) -> BiharmonicNasalResult:
    """Fit a smooth scalar normal-displacement field with a fixed outer ring."""
    limits = config or BiharmonicNasalConfig()
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    face_indices = np.asarray(attachment_face_indices, dtype=np.int64).reshape(-1)
    barycentric = np.asarray(attachment_barycentric, dtype=np.float64)
    targets = np.asarray(target_points_model, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64).reshape(-1)
    reference = (
        np.zeros(len(baseline), dtype=np.float64)
        if reference_scalar_displacement_m is None
        else np.asarray(reference_scalar_displacement_m, dtype=np.float64).reshape(-1)
    )
    count = len(face_indices)
    if baseline.ndim != 2 or baseline.shape[1] != 3 or not np.isfinite(baseline).all():
        raise ValueError("baseline_vertices must have finite shape (V, 3)")
    if topology.ndim != 2 or topology.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if reference.shape != (len(baseline),) or not np.isfinite(reference).all():
        raise ValueError("reference displacement must match baseline vertices")
    if support.shape != (len(baseline),) or np.count_nonzero(support) < 9:
        raise ValueError("support_mask must select at least nine vertices")
    if (
        barycentric.shape != (count, 3)
        or targets.shape != (count, 3)
        or weights.shape != (count,)
        or count < 4
        or np.any(face_indices < 0)
        or np.any(face_indices >= len(topology))
        or not np.isfinite(barycentric).all()
        or not np.allclose(np.sum(barycentric, axis=1), 1.0, atol=1e-6)
        or not np.isfinite(targets).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("surface observations are invalid")

    fixed = fixed_inner_boundary_mask(
        support,
        topology,
        rings=int(limits.fixed_boundary_rings),
    )
    free = np.flatnonzero(support & ~fixed)
    free_index = np.full(len(baseline), -1, dtype=np.int64)
    free_index[free] = np.arange(len(free), dtype=np.int64)
    if len(free) < 4:
        raise ValueError("support has too few free vertices after fixing its boundary")

    triangles = topology[face_indices]
    usable = np.all(support[triangles], axis=1) & np.any(
        free_index[triangles] >= 0,
        axis=1,
    )
    if np.count_nonzero(usable) < 4:
        raise ValueError("fewer than four observations attach to the free support")
    triangles = triangles[usable]
    bary = barycentric[usable]
    target = targets[usable]
    used_weights = weights[usable]
    baseline_points = np.sum(baseline[triangles] * bary[:, :, None], axis=1)
    nuisance = _weighted_median(target - baseline_points, used_weights)
    desired = target - nuisance - baseline_points

    normals = _vertex_normals(baseline, topology)
    data_rows: list[int] = []
    data_columns: list[int] = []
    data_values: list[float] = []
    data_rhs = np.zeros(3 * len(triangles), dtype=np.float64)
    for observation_index, (triangle, coordinates) in enumerate(zip(triangles, bary)):
        for axis in range(3):
            row = 3 * observation_index + axis
            data_rhs[row] = desired[observation_index, axis]
            for vertex, coordinate in zip(triangle, coordinates):
                column = int(free_index[int(vertex)])
                if column >= 0:
                    data_rows.append(row)
                    data_columns.append(column)
                    data_values.append(
                        float(coordinate) * float(normals[int(vertex), axis])
                    )
    data_matrix = sparse.coo_matrix(
        (data_values, (data_rows, data_columns)),
        shape=(len(data_rhs), len(free)),
        dtype=np.float64,
    ).tocsr()

    adjacency = _adjacency(len(baseline), topology)
    lap_rows: list[int] = []
    lap_columns: list[int] = []
    lap_values: list[float] = []
    for row, vertex in enumerate(free):
        neighbors = adjacency[int(vertex)]
        if not len(neighbors):
            continue
        lap_rows.append(row)
        lap_columns.append(row)
        lap_values.append(1.0)
        inverse_degree = 1.0 / float(len(neighbors))
        for neighbor in neighbors:
            column = int(free_index[int(neighbor)])
            if column >= 0:
                lap_rows.append(row)
                lap_columns.append(column)
                lap_values.append(-inverse_degree)
    laplacian = sparse.coo_matrix(
        (lap_values, (lap_rows, lap_columns)),
        shape=(len(free), len(free)),
        dtype=np.float64,
    ).tocsr()
    regularization_scale = np.sqrt(
        float(len(data_rhs)) / max(float(len(free)), 1.0)
    )
    bending_matrix = (
        np.sqrt(float(limits.bending_weight))
        * regularization_scale
        * laplacian
    )
    prior_matrix = (
        np.sqrt(float(limits.displacement_prior_weight))
        * regularization_scale
        * sparse.eye(len(free), format="csr")
    )
    prior_rhs = (
        np.sqrt(float(limits.displacement_prior_weight))
        * regularization_scale
        * reference[free]
    )
    normalized_weights = used_weights / max(float(np.median(used_weights)), 1e-12)
    robust_weights = np.ones(len(triangles), dtype=np.float64)
    solved = None
    scalar_free = np.zeros(len(free), dtype=np.float64)
    robust_delta_m = float(limits.robust_delta_mm) / 1000.0
    for _ in range(int(limits.robust_iterations)):
        row_scale = np.repeat(
            np.sqrt(normalized_weights * robust_weights),
            3,
        )
        weighted_data = sparse.diags(row_scale, format="csr") @ data_matrix
        system = sparse.vstack(
            (weighted_data, bending_matrix, prior_matrix),
            format="csr",
        )
        rhs = np.r_[
            data_rhs * row_scale,
            np.zeros(len(free), dtype=np.float64),
            prior_rhs,
        ]
        solved = lsqr(
            system,
            rhs,
            atol=float(limits.lsqr_tolerance),
            btol=float(limits.lsqr_tolerance),
            iter_lim=int(limits.max_iterations),
        )
        scalar_free = np.asarray(solved[0], dtype=np.float64)
        residual_vectors = (data_matrix @ scalar_free - data_rhs).reshape(-1, 3)
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        next_robust = np.minimum(
            1.0,
            robust_delta_m / np.maximum(residual_norms, 1e-12),
        )
        if np.max(np.abs(next_robust - robust_weights)) <= 1e-3:
            robust_weights = next_robust
            break
        robust_weights = next_robust
    assert solved is not None
    scalar = np.zeros(len(baseline), dtype=np.float64)
    scalar[free] = scalar_free
    candidate = baseline + normals * scalar[:, None]
    predicted = np.sum(candidate[triangles] * bary[:, :, None], axis=1) + nuisance
    initial_rmse = float(
        np.sqrt(np.average(np.sum(desired**2, axis=1), weights=used_weights))
        * 1000.0
    )
    final_rmse = float(
        np.sqrt(
            np.average(np.sum((predicted - target) ** 2, axis=1), weights=used_weights)
        )
        * 1000.0
    )
    return BiharmonicNasalResult(
        candidate_vertices=candidate,
        scalar_displacement_m=scalar,
        support_mask=support,
        fixed_mask=fixed,
        observation_mask=usable,
        initial_rmse_mm=initial_rmse,
        final_rmse_mm=final_rmse,
        nuisance_translation_mm=nuisance * 1000.0,
        metadata={
            "mode": "fixed_double_ring_normal_biharmonic",
            "bending_weight": float(limits.bending_weight),
            "displacement_prior_weight": float(limits.displacement_prior_weight),
            "displacement_prior_target": (
                "zero" if reference_scalar_displacement_m is None else "reference"
            ),
            "robust_delta_mm": float(limits.robust_delta_mm),
            "robust_iterations": int(limits.robust_iterations),
            "downweighted_observation_count": int(
                np.count_nonzero(robust_weights < 0.999)
            ),
            "minimum_robust_weight": float(np.min(robust_weights)),
            "median_robust_weight": float(np.median(robust_weights)),
            "lsqr_stop_code": int(solved[1]),
            "lsqr_iterations": int(solved[2]),
            "lsqr_residual_norm": float(solved[3]),
            "maximum_displacement_mm": float(np.max(np.abs(scalar)) * 1000.0),
            "displacement_p95_mm": float(
                np.percentile(np.abs(scalar[free]), 95.0) * 1000.0
            ),
            "bending_rms_mm": float(
                np.sqrt(np.mean(np.asarray(laplacian @ scalar[free]) ** 2))
                * 1000.0
            ),
        },
    )


def fit_vector_biharmonic_nasal_surface(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    support_mask: np.ndarray,
    attachment_face_indices: np.ndarray,
    attachment_barycentric: np.ndarray,
    target_points_model: np.ndarray,
    observation_weights: np.ndarray,
    *,
    reference_displacement_m: np.ndarray | None = None,
    config: BiharmonicNasalConfig | None = None,
) -> VectorBiharmonicNasalResult:
    """Fit a smooth three-dimensional displacement field with a fixed boundary."""
    limits = config or BiharmonicNasalConfig()
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    support = np.asarray(support_mask, dtype=bool).reshape(-1)
    face_indices = np.asarray(attachment_face_indices, dtype=np.int64).reshape(-1)
    barycentric = np.asarray(attachment_barycentric, dtype=np.float64)
    targets = np.asarray(target_points_model, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64).reshape(-1)
    reference = (
        np.zeros_like(baseline)
        if reference_displacement_m is None
        else np.asarray(reference_displacement_m, dtype=np.float64)
    )
    count = len(face_indices)
    if baseline.ndim != 2 or baseline.shape[1] != 3 or not np.isfinite(baseline).all():
        raise ValueError("baseline_vertices must have finite shape (V, 3)")
    if topology.ndim != 2 or topology.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if reference.shape != baseline.shape or not np.isfinite(reference).all():
        raise ValueError("reference displacement must match baseline vertices")
    if support.shape != (len(baseline),) or np.count_nonzero(support) < 9:
        raise ValueError("support_mask must select at least nine vertices")
    if (
        barycentric.shape != (count, 3)
        or targets.shape != (count, 3)
        or weights.shape != (count,)
        or count < 4
        or np.any(face_indices < 0)
        or np.any(face_indices >= len(topology))
        or not np.isfinite(barycentric).all()
        or not np.allclose(np.sum(barycentric, axis=1), 1.0, atol=1e-6)
        or not np.isfinite(targets).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("surface observations are invalid")

    fixed = fixed_inner_boundary_mask(
        support,
        topology,
        rings=int(limits.fixed_boundary_rings),
    )
    free = np.flatnonzero(support & ~fixed)
    free_index = np.full(len(baseline), -1, dtype=np.int64)
    free_index[free] = np.arange(len(free), dtype=np.int64)
    if len(free) < 4:
        raise ValueError("support has too few free vertices after fixing its boundary")

    triangles = topology[face_indices]
    usable = np.all(support[triangles], axis=1) & np.any(
        free_index[triangles] >= 0,
        axis=1,
    )
    if np.count_nonzero(usable) < 4:
        raise ValueError("fewer than four observations attach to the free support")
    triangles = triangles[usable]
    bary = barycentric[usable]
    target = targets[usable]
    used_weights = weights[usable]
    baseline_points = np.sum(baseline[triangles] * bary[:, :, None], axis=1)
    nuisance = _weighted_median(target - baseline_points, used_weights)
    desired = target - nuisance - baseline_points

    data_rows: list[int] = []
    data_columns: list[int] = []
    data_values: list[float] = []
    for observation_index, (triangle, coordinates) in enumerate(zip(triangles, bary)):
        for vertex, coordinate in zip(triangle, coordinates):
            column = int(free_index[int(vertex)])
            if column >= 0:
                data_rows.append(observation_index)
                data_columns.append(column)
                data_values.append(float(coordinate))
    data_matrix = sparse.coo_matrix(
        (data_values, (data_rows, data_columns)),
        shape=(len(triangles), len(free)),
        dtype=np.float64,
    ).tocsr()

    adjacency = _adjacency(len(baseline), topology)
    lap_rows: list[int] = []
    lap_columns: list[int] = []
    lap_values: list[float] = []
    for row, vertex in enumerate(free):
        neighbors = adjacency[int(vertex)]
        if not len(neighbors):
            continue
        lap_rows.append(row)
        lap_columns.append(row)
        lap_values.append(1.0)
        inverse_degree = 1.0 / float(len(neighbors))
        for neighbor in neighbors:
            column = int(free_index[int(neighbor)])
            if column >= 0:
                lap_rows.append(row)
                lap_columns.append(column)
                lap_values.append(-inverse_degree)
    laplacian = sparse.coo_matrix(
        (lap_values, (lap_rows, lap_columns)),
        shape=(len(free), len(free)),
        dtype=np.float64,
    ).tocsr()
    regularization_scale_sq = float(len(triangles)) / max(float(len(free)), 1.0)
    bending_normal = (
        float(limits.bending_weight)
        * regularization_scale_sq
        * (laplacian.T @ laplacian)
    )
    prior_strength = float(limits.displacement_prior_weight) * regularization_scale_sq
    prior_normal = prior_strength * sparse.eye(len(free), format="csr")

    normalized_weights = used_weights / max(float(np.median(used_weights)), 1e-12)
    robust_weights = np.ones(len(triangles), dtype=np.float64)
    solved_free = np.zeros((len(free), 3), dtype=np.float64)
    robust_delta_m = float(limits.robust_delta_mm) / 1000.0
    robust_steps = 0
    for robust_steps in range(1, int(limits.robust_iterations) + 1):
        precision = normalized_weights * robust_weights
        weighted_data = sparse.diags(precision, format="csr")
        normal_matrix = (
            data_matrix.T @ weighted_data @ data_matrix
            + bending_normal
            + prior_normal
        ).tocsc()
        normal_rhs = data_matrix.T @ (precision[:, None] * desired)
        normal_rhs += prior_strength * reference[free]
        solved_free = np.asarray(spsolve(normal_matrix, normal_rhs), dtype=np.float64)
        if solved_free.shape != (len(free), 3) or not np.isfinite(solved_free).all():
            raise RuntimeError("vector biharmonic solve returned invalid displacement")
        residual_vectors = data_matrix @ solved_free - desired
        residual_norms = np.linalg.norm(residual_vectors, axis=1)
        next_robust = np.minimum(
            1.0,
            robust_delta_m / np.maximum(residual_norms, 1e-12),
        )
        if np.max(np.abs(next_robust - robust_weights)) <= 1e-3:
            robust_weights = next_robust
            break
        robust_weights = next_robust

    displacement = np.zeros_like(baseline)
    displacement[free] = solved_free
    candidate = baseline + displacement
    predicted = np.sum(candidate[triangles] * bary[:, :, None], axis=1) + nuisance
    initial_rmse = float(
        np.sqrt(np.average(np.sum(desired**2, axis=1), weights=used_weights))
        * 1000.0
    )
    final_rmse = float(
        np.sqrt(
            np.average(np.sum((predicted - target) ** 2, axis=1), weights=used_weights)
        )
        * 1000.0
    )
    displacement_norm = np.linalg.norm(displacement[free], axis=1)
    bending_norm = np.linalg.norm(np.asarray(laplacian @ solved_free), axis=1)
    return VectorBiharmonicNasalResult(
        candidate_vertices=candidate,
        displacement_m=displacement,
        support_mask=support,
        fixed_mask=fixed,
        observation_mask=usable,
        initial_rmse_mm=initial_rmse,
        final_rmse_mm=final_rmse,
        nuisance_translation_mm=nuisance * 1000.0,
        metadata={
            "mode": "fixed_double_ring_vector_biharmonic",
            "bending_weight": float(limits.bending_weight),
            "displacement_prior_weight": float(limits.displacement_prior_weight),
            "displacement_prior_target": (
                "zero" if reference_displacement_m is None else "reference"
            ),
            "robust_delta_mm": float(limits.robust_delta_mm),
            "robust_iterations": int(robust_steps),
            "downweighted_observation_count": int(
                np.count_nonzero(robust_weights < 0.999)
            ),
            "minimum_robust_weight": float(np.min(robust_weights)),
            "median_robust_weight": float(np.median(robust_weights)),
            "solver": "sparse_normal_equations_3d",
            "maximum_displacement_mm": float(np.max(displacement_norm) * 1000.0),
            "displacement_p95_mm": float(
                np.percentile(displacement_norm, 95.0) * 1000.0
            ),
            "bending_rms_mm": float(
                np.sqrt(np.mean(bending_norm**2)) * 1000.0
            ),
        },
    )


__all__ = [
    "BiharmonicNasalConfig",
    "BiharmonicNasalResult",
    "VectorBiharmonicNasalResult",
    "expand_vertex_mask",
    "fit_biharmonic_nasal_surface",
    "fit_vector_biharmonic_nasal_surface",
    "fixed_inner_boundary_mask",
    "project_normal_displacement",
]

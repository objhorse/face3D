"""Fixed-topology semantic control bands for both eyelids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np
from scipy import sparse

from src.geometry.semantic_regions import (
    _graph_distances,
    _mesh_adjacency,
    apply_semantic_control_offsets,
    build_semantic_control_weights,
)


@dataclass(frozen=True)
class SemanticEyelidRig:
    control_names: tuple[str, ...]
    weights: np.ndarray
    active_vertices: np.ndarray
    protected_vertices: np.ndarray
    core_vertices: np.ndarray
    transition_vertices: np.ndarray
    control_seeds: Mapping[str, np.ndarray]
    vertex_normals: np.ndarray
    control_normals: np.ndarray

    def apply(self, vertices: np.ndarray, control_offsets: np.ndarray) -> np.ndarray:
        return apply_semantic_control_offsets(vertices, self.weights, control_offsets)

    def to_dict(self) -> dict:
        return {
            "control_names": list(self.control_names),
            "active_vertices": self.active_vertices.astype(int).tolist(),
            "protected_vertices": self.protected_vertices.astype(int).tolist(),
            "core_vertices": self.core_vertices.astype(int).tolist(),
            "transition_vertices": self.transition_vertices.astype(int).tolist(),
            "control_seeds": {
                name: np.asarray(indices, dtype=np.int64).astype(int).tolist()
                for name, indices in self.control_seeds.items()
            },
            "control_normals": self.control_normals.astype(float).tolist(),
        }


def _seed_vertices(triangles: np.ndarray, indices: tuple[int, ...]) -> np.ndarray:
    return np.unique(triangles[np.asarray(indices, dtype=np.int64)].reshape(-1))


def _vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    vertices_f = np.asarray(vertices, dtype=np.float64)
    faces_i = np.asarray(faces, dtype=np.int64)
    normals = np.zeros_like(vertices_f)
    tri = vertices_f[faces_i]
    face_normals = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    for corner in range(3):
        np.add.at(normals, faces_i[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals /= np.maximum(lengths, 1e-12)
    return normals.astype(np.float32)


def _length_weighted_smooth_control_weights(
    vertices: np.ndarray,
    faces: np.ndarray,
    weights: np.ndarray,
    active_vertices: np.ndarray,
    *,
    iterations: int,
    retention: float,
) -> np.ndarray:
    """Screen graph-ring weights with a physical edge-length Laplacian."""
    if iterations <= 0:
        return np.asarray(weights, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    undirected = np.vstack(
        (faces_i[:, [0, 1]], faces_i[:, [1, 2]], faces_i[:, [2, 0]])
    )
    undirected.sort(axis=1)
    undirected = np.unique(undirected, axis=0)
    vertices_f = np.asarray(vertices, dtype=np.float64)
    lengths = np.linalg.norm(
        vertices_f[undirected[:, 1]] - vertices_f[undirected[:, 0]],
        axis=1,
    )
    positive = lengths[lengths > 1e-12]
    reference = float(np.median(positive)) if len(positive) else 1.0
    safe_lengths = np.maximum(lengths, max(reference * 0.02, 1e-12))
    conductance = 1.0 / np.square(safe_lengths)
    rows = np.concatenate((undirected[:, 0], undirected[:, 1]))
    columns = np.concatenate((undirected[:, 1], undirected[:, 0]))
    data = np.concatenate((conductance, conductance))
    adjacency = sparse.csr_matrix(
        (data, (rows, columns)),
        shape=(len(vertices_f), len(vertices_f)),
        dtype=np.float64,
    )
    row_sum = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    row_sum = np.maximum(row_sum, 1e-12)
    source = np.asarray(weights, dtype=np.float64)
    smoothed = source.copy()
    active_mask = np.zeros(len(vertices_f), dtype=bool)
    active_mask[np.asarray(active_vertices, dtype=np.int64)] = True
    keep = float(np.clip(retention, 0.0, 1.0))
    relaxation = 0.5
    for _ in range(int(iterations)):
        neighbor_average = adjacency @ smoothed
        neighbor_average /= row_sum[:, None]
        target = (
            keep * source[active_mask]
            + (1.0 - keep) * neighbor_average[active_mask]
        )
        smoothed[active_mask] = (
            (1.0 - relaxation) * smoothed[active_mask]
            + relaxation * target
        )
        smoothed[~active_mask] = 0.0
    return np.maximum(smoothed, 0.0).astype(np.float32)


def build_semantic_eyelid_rig(
    vertices: np.ndarray,
    faces: np.ndarray,
    lmk_tri_vidx: np.ndarray,
    *,
    support_rings: int = 8,
    core_rings: int = 2,
    sigma_rings: float = 3.0,
    smoothing_iterations: int = 40,
    smoothing_retention: float = 0.03,
    freeze_corner_seeds: bool = False,
) -> SemanticEyelidRig:
    vertices_f = np.asarray(vertices, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    triangles = np.asarray(lmk_tri_vidx, dtype=np.int64)
    if triangles.shape[0] < 68 or triangles.shape[1] != 3:
        raise ValueError("lmk_tri_vidx must have shape (68, 3)")
    control_seeds = {
        "subject_right_outer_corner": _seed_vertices(triangles, (36,)),
        "subject_right_upper_lid": _seed_vertices(triangles, (37, 38)),
        "subject_right_inner_corner": _seed_vertices(triangles, (39,)),
        "subject_right_lower_lid": _seed_vertices(triangles, (40, 41)),
        "subject_left_inner_corner": _seed_vertices(triangles, (42,)),
        "subject_left_upper_lid": _seed_vertices(triangles, (43, 44)),
        "subject_left_outer_corner": _seed_vertices(triangles, (45,)),
        "subject_left_lower_lid": _seed_vertices(triangles, (46, 47)),
    }
    names, weights, active = build_semantic_control_weights(
        faces_i,
        control_seeds,
        len(vertices_f),
        support_rings=int(support_rings),
        sigma_rings=float(sigma_rings),
    )
    weights = _length_weighted_smooth_control_weights(
        vertices_f,
        faces_i,
        weights,
        active,
        iterations=int(smoothing_iterations),
        retention=float(smoothing_retention),
    )
    corner_names = tuple(
        name for name in names if name.endswith("_corner")
    )
    protected = (
        np.unique(
            np.concatenate([control_seeds[name] for name in corner_names])
        ).astype(np.int64)
        if corner_names
        else np.empty(0, dtype=np.int64)
    )
    if freeze_corner_seeds and len(protected):
        weights[protected] = 0.0
        active = np.flatnonzero(np.any(weights > 0.0, axis=1)).astype(np.int64)
    adjacency = _mesh_adjacency(faces_i, len(vertices_f))
    all_seeds = np.unique(np.concatenate(list(control_seeds.values())))
    distances = _graph_distances(adjacency, all_seeds, int(support_rings))
    core = np.flatnonzero((distances >= 0) & (distances <= int(core_rings))).astype(np.int64)
    transition = np.setdiff1d(active, core, assume_unique=False).astype(np.int64)
    normals = _vertex_normals(vertices_f, faces_i)
    control_normals = []
    for name in names:
        normal = normals[control_seeds[name]].mean(axis=0)
        length = float(np.linalg.norm(normal))
        if length <= 1e-8:
            normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
        else:
            normal = normal / length
        control_normals.append(normal)
    return SemanticEyelidRig(
        control_names=tuple(names),
        weights=weights.astype(np.float32),
        active_vertices=active.astype(np.int64),
        protected_vertices=protected,
        core_vertices=core,
        transition_vertices=transition,
        control_seeds={key: value.astype(np.int64) for key, value in control_seeds.items()},
        vertex_normals=normals,
        control_normals=np.asarray(control_normals, dtype=np.float32),
    )

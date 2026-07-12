from __future__ import annotations

from collections import deque
from typing import Iterable, Mapping

import numpy as np


def _mesh_adjacency(faces: np.ndarray, n_vertices: int) -> list[list[int]]:
    adjacency = [set() for _ in range(int(n_vertices))]
    for a, b, c in np.asarray(faces, dtype=np.int64):
        adjacency[int(a)].update((int(b), int(c)))
        adjacency[int(b)].update((int(a), int(c)))
        adjacency[int(c)].update((int(a), int(b)))
    return [sorted(neighbors) for neighbors in adjacency]


def _graph_distances(
    adjacency: list[list[int]],
    seeds: Iterable[int],
    max_distance: int,
) -> np.ndarray:
    distances = np.full(len(adjacency), -1, dtype=np.int32)
    queue: deque[int] = deque()
    for seed in np.unique(np.asarray(list(seeds), dtype=np.int64)):
        if 0 <= int(seed) < len(adjacency):
            distances[int(seed)] = 0
            queue.append(int(seed))
    while queue:
        vertex = queue.popleft()
        distance = int(distances[vertex])
        if distance >= int(max_distance):
            continue
        for neighbor in adjacency[vertex]:
            if distances[neighbor] < 0:
                distances[neighbor] = distance + 1
                queue.append(neighbor)
    return distances


def build_semantic_control_weights(
    faces: np.ndarray,
    control_seeds: Mapping[str, Iterable[int]],
    n_vertices: int,
    support_rings: int = 6,
    sigma_rings: float = 2.5,
) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Create smooth, compact-support vertex weights for semantic controls."""
    if not control_seeds:
        raise ValueError("control_seeds is empty")
    names = list(control_seeds.keys())
    adjacency = _mesh_adjacency(faces, n_vertices)
    weights = np.zeros((int(n_vertices), len(names)), dtype=np.float32)
    sigma = max(float(sigma_rings), 1e-3)
    distance_columns = []
    for column, name in enumerate(names):
        distances = _graph_distances(adjacency, control_seeds[name], int(support_rings))
        distance_columns.append(distances)
        supported = distances >= 0
        weights[supported, column] = np.exp(
            -0.5 * (distances[supported].astype(np.float32) / sigma) ** 2
        )
    total = weights.sum(axis=1, keepdims=True)
    active = np.flatnonzero(total[:, 0] > 0.0).astype(np.int64)
    weights[active] /= total[active]
    min_distance = np.min(
        np.stack([np.where(d >= 0, d, support_rings + 1) for d in distance_columns], axis=1),
        axis=1,
    )
    envelope = np.clip(
        (float(support_rings) + 1.0 - min_distance.astype(np.float32))
        / (float(support_rings) + 1.0),
        0.0,
        1.0,
    )
    weights *= envelope[:, None]
    return names, weights, active


def build_default_nose_mouth_control_seeds(lmk_tri_vidx: np.ndarray) -> dict[str, np.ndarray]:
    """Create subject-relative semantic control seeds from the 68-point mapping."""
    triangles = np.asarray(lmk_tri_vidx, dtype=np.int64)
    if triangles.shape[0] < 68 or triangles.shape[1] != 3:
        raise ValueError("lmk_tri_vidx must have shape (68, 3)")

    def vertices(indices: Iterable[int]) -> np.ndarray:
        return np.unique(triangles[np.asarray(list(indices), dtype=np.int64)].reshape(-1))

    return {
        "nose_bridge": vertices([27, 28, 29]),
        "nose_tip": vertices([30, 33]),
        "subject_right_nose_wing": vertices([31, 32]),
        "subject_left_nose_wing": vertices([34, 35]),
        "philtrum": vertices([33, 51, 62]),
        "subject_right_mouth_corner": vertices([48, 60]),
        "upper_lip": vertices([49, 50, 51, 52, 53, 61, 62, 63]),
        "subject_left_mouth_corner": vertices([54, 64]),
        "lower_lip": vertices([55, 56, 57, 58, 59, 65, 66, 67]),
    }


def apply_semantic_control_offsets(
    vertices: np.ndarray,
    weights: np.ndarray,
    control_offsets: np.ndarray,
) -> np.ndarray:
    vertices_np = np.asarray(vertices, dtype=np.float32)
    weights_np = np.asarray(weights, dtype=np.float32)
    controls_np = np.asarray(control_offsets, dtype=np.float32)
    if weights_np.shape[0] != len(vertices_np):
        raise ValueError("weights and vertices have incompatible shapes")
    if weights_np.shape[1] != len(controls_np) or controls_np.shape[1] != 3:
        raise ValueError("control offsets must have shape (controls, 3)")
    return vertices_np + weights_np @ controls_np

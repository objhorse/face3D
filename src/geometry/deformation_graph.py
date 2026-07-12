"""Low-frequency, surface-connected deformation controls for fixed topology meshes."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class DeformationGraph:
    node_vertex_indices: np.ndarray
    node_regions: tuple[str, ...]
    vertex_node_indices: np.ndarray
    vertex_node_weights: np.ndarray
    node_edges: np.ndarray


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    edges = np.vstack((triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _surface_graph(vertices: np.ndarray, faces: np.ndarray):
    from scipy.sparse import coo_matrix

    verts = np.asarray(vertices, dtype=np.float64)
    edges = _mesh_edges(faces)
    lengths = np.linalg.norm(verts[edges[:, 1]] - verts[edges[:, 0]], axis=1)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.concatenate((lengths, lengths))
    return coo_matrix((data, (rows, cols)), shape=(len(verts), len(verts))).tocsr()


def _normalize_regions(region_labels: Sequence[str] | Mapping[int, str], n_vertices: int) -> np.ndarray:
    regions = np.full(n_vertices, "face", dtype=object)
    if isinstance(region_labels, Mapping):
        for index, label in region_labels.items():
            if 0 <= int(index) < n_vertices:
                regions[int(index)] = str(label)
    else:
        labels = np.asarray(region_labels, dtype=object).reshape(-1)
        if len(labels) != n_vertices:
            raise ValueError("region_labels must contain one label per vertex")
        regions[:] = labels
    return regions.astype(str)


def _farthest_surface_nodes(graph, allowed: np.ndarray, regions: np.ndarray, count: int) -> np.ndarray:
    from scipy.sparse.csgraph import dijkstra

    candidates = np.flatnonzero(allowed)
    if not len(candidates):
        raise ValueError("allowed_vertices is empty")
    count = min(int(count), len(candidates))
    chosen: list[int] = []
    # Seed each known region once so nose/chin cannot disappear behind large cheeks.
    for region in sorted(set(regions[candidates])):
        regional = candidates[regions[candidates] == region]
        if len(regional) and len(chosen) < count:
            chosen.append(int(regional[0]))
    chosen = chosen[:count]
    if not chosen:
        chosen = [int(candidates[0])]
    minimum_distance = np.full(graph.shape[0], np.inf, dtype=np.float64)
    for node in chosen:
        distance = dijkstra(graph, indices=node, directed=False)
        minimum_distance = np.minimum(minimum_distance, distance)
    while len(chosen) < count:
        available = candidates[~np.isin(candidates, np.asarray(chosen, dtype=np.int64))]
        next_node = int(available[np.argmax(minimum_distance[available])])
        chosen.append(next_node)
        distance = dijkstra(graph, indices=next_node, directed=False)
        minimum_distance = np.minimum(minimum_distance, distance)
    return np.asarray(chosen, dtype=np.int64)


def build_deformation_graph(
    vertices: np.ndarray,
    faces: np.ndarray,
    allowed_vertices: Sequence[int],
    region_labels: Sequence[str] | Mapping[int, str],
    node_count: int = 220,
    influences: int = 6,
) -> DeformationGraph:
    """Sample surface nodes and derive compact, geodesic vertex influences."""
    from scipy.sparse.csgraph import dijkstra

    verts = np.asarray(vertices, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3:
        raise ValueError("vertices must have shape (N, 3)")
    allowed = np.zeros(len(verts), dtype=bool)
    indices = np.unique(np.asarray(allowed_vertices, dtype=np.int64))
    indices = indices[(indices >= 0) & (indices < len(verts))]
    allowed[indices] = True
    regions = _normalize_regions(region_labels, len(verts))
    graph = _surface_graph(verts, faces)
    nodes = _farthest_surface_nodes(graph, allowed, regions, node_count)
    distances = dijkstra(graph, indices=nodes, directed=False)
    influences = min(max(int(influences), 1), len(nodes))
    nearest_order = np.argsort(distances, axis=0)[:influences].T
    nearest_distances = np.take_along_axis(distances.T, nearest_order, axis=1)
    finite = np.isfinite(nearest_distances)
    weights = np.zeros_like(nearest_distances, dtype=np.float64)
    positive = nearest_distances[finite & (nearest_distances > 1e-12)]
    sigma = float(np.percentile(positive, 70)) if len(positive) else 1.0
    sigma = max(sigma, 1e-8)
    weights[finite] = np.exp(-0.5 * (nearest_distances[finite] / sigma) ** 2)
    weights[~allowed] = 0.0
    sums = weights.sum(axis=1, keepdims=True)
    active = sums[:, 0] > 1e-12
    weights[active] /= sums[active]
    nearest_order[~active] = -1

    node_distances = distances[:, nodes]
    edge_pairs = []
    for node_index in range(len(nodes)):
        candidates = np.argsort(node_distances[node_index])
        for other in candidates:
            if other == node_index or not np.isfinite(node_distances[node_index, other]):
                continue
            pair = tuple(sorted((node_index, int(other))))
            edge_pairs.append(pair)
            if sum(1 for item in edge_pairs if node_index in item) >= 4:
                break
    node_edges = np.unique(np.asarray(edge_pairs, dtype=np.int64), axis=0) if edge_pairs else np.empty((0, 2), dtype=np.int64)
    return DeformationGraph(
        node_vertex_indices=nodes,
        node_regions=tuple(regions[nodes].tolist()),
        vertex_node_indices=nearest_order.astype(np.int64),
        vertex_node_weights=weights.astype(np.float32),
        node_edges=node_edges,
    )


def apply_node_translations(vertices: np.ndarray, graph: DeformationGraph, translations: np.ndarray) -> np.ndarray:
    verts = np.asarray(vertices, dtype=np.float32)
    controls = np.asarray(translations, dtype=np.float32)
    if controls.shape != (len(graph.node_vertex_indices), 3):
        raise ValueError("translations must have shape (node_count, 3)")
    delta = np.zeros_like(verts)
    for column in range(graph.vertex_node_indices.shape[1]):
        node_index = graph.vertex_node_indices[:, column]
        valid = node_index >= 0
        if np.any(valid):
            delta[valid] += graph.vertex_node_weights[valid, column, None] * controls[node_index[valid]]
    return verts + delta


def smooth_vertex_displacements(
    faces: np.ndarray,
    displacements: np.ndarray,
    iterations: int = 20,
    relaxation: float = 0.5,
) -> np.ndarray:
    """Remove control-cell seams while retaining the low-frequency displacement."""
    from scipy.sparse import coo_matrix

    delta = np.asarray(displacements, dtype=np.float64).copy()
    edges = _mesh_edges(faces)
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    cols = np.concatenate((edges[:, 1], edges[:, 0]))
    data = np.ones(len(rows), dtype=np.float64)
    adjacency = coo_matrix((data, (rows, cols)), shape=(len(delta), len(delta))).tocsr()
    degree = np.asarray(adjacency.sum(axis=1)).reshape(-1)
    active = degree > 0
    amount = float(np.clip(relaxation, 0.0, 1.0))
    for _ in range(max(int(iterations), 0)):
        neighbor_mean = adjacency @ delta
        neighbor_mean[active] /= degree[active, None]
        delta[active] = (1.0 - amount) * delta[active] + amount * neighbor_mean[active]
    return delta.astype(np.float32)

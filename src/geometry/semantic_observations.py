from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
import trimesh


@dataclass(frozen=True)
class TriangleAttachments:
    face_indices: np.ndarray
    bary_coords: np.ndarray
    surface_points: np.ndarray
    target_points: np.ndarray
    distances: np.ndarray


def attach_points_to_active_mesh(
    points: np.ndarray,
    vertices: np.ndarray,
    faces: np.ndarray,
    active_vertices: Iterable[int],
) -> TriangleAttachments:
    points_np = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    vertices_np = np.asarray(vertices, dtype=np.float64)
    faces_np = np.asarray(faces, dtype=np.int64)
    active = np.zeros(len(vertices_np), dtype=bool)
    active_idx = np.asarray(list(active_vertices), dtype=np.int64)
    active_idx = active_idx[(active_idx >= 0) & (active_idx < len(active))]
    active[active_idx] = True
    face_mask = np.any(active[faces_np], axis=1)
    selected_ids = np.flatnonzero(face_mask)
    if len(points_np) == 0 or len(selected_ids) == 0:
        return TriangleAttachments(
            face_indices=np.empty(0, dtype=np.int64),
            bary_coords=np.empty((0, 3), dtype=np.float32),
            surface_points=np.empty((0, 3), dtype=np.float32),
            target_points=points_np.astype(np.float32),
            distances=np.empty(0, dtype=np.float32),
        )

    selected_faces = faces_np[selected_ids]
    mesh = trimesh.Trimesh(vertices=vertices_np, faces=selected_faces, process=False)
    surface, distances, local_face_ids = trimesh.proximity.closest_point_naive(mesh, points_np)
    global_face_ids = selected_ids[np.asarray(local_face_ids, dtype=np.int64)]
    triangles = vertices_np[faces_np[global_face_ids]]
    bary = trimesh.triangles.points_to_barycentric(triangles, surface)
    return TriangleAttachments(
        face_indices=global_face_ids.astype(np.int64),
        bary_coords=np.asarray(bary, dtype=np.float32),
        surface_points=np.asarray(surface, dtype=np.float32),
        target_points=points_np.astype(np.float32),
        distances=np.asarray(distances, dtype=np.float32),
    )


def semantic_boundary_vertices(
    faces: np.ndarray,
    region_vertices: Iterable[int],
    n_vertices: int,
) -> np.ndarray:
    region = np.zeros(int(n_vertices), dtype=bool)
    indices = np.asarray(list(region_vertices), dtype=np.int64)
    indices = indices[(indices >= 0) & (indices < len(region))]
    region[indices] = True
    has_outside_neighbor = np.zeros(int(n_vertices), dtype=bool)
    for face in np.asarray(faces, dtype=np.int64):
        values = region[face]
        if np.any(values) and not np.all(values):
            has_outside_neighbor[face[values]] = True
    return np.flatnonzero(region & has_outside_neighbor).astype(np.int64)


def mask_boundary_distance(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
    if mask is None:
        return None
    binary = (np.asarray(mask) > 0).astype(np.uint8)
    if int(binary.sum()) == 0:
        return None
    kernel = np.ones((3, 3), dtype=np.uint8)
    boundary = cv2.morphologyEx(binary, cv2.MORPH_GRADIENT, kernel) > 0
    inverse = (~boundary).astype(np.uint8)
    return cv2.distanceTransform(inverse, cv2.DIST_L2, 3).astype(np.float32)

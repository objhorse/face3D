from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class LocalMeshQualityConfig:
    max_edge_ratio: float = 3.0
    min_area_ratio: float = 0.05
    max_normal_angle_deg: float = 85.0
    absolute_min_area: float = 1e-12


def _face_geometry(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    tri = vertices[faces]
    cross = np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0])
    norm = np.linalg.norm(cross, axis=1)
    unit = cross / np.maximum(norm[:, None], 1e-15)
    return 0.5 * norm, unit


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _active_face_mask(faces: np.ndarray, active_vertices: Optional[np.ndarray]) -> np.ndarray:
    if active_vertices is None:
        return np.ones(len(faces), dtype=bool)
    active = np.zeros(int(faces.max()) + 1, dtype=bool)
    clipped = np.asarray(active_vertices, dtype=np.int64)
    clipped = clipped[(clipped >= 0) & (clipped < len(active))]
    active[clipped] = True
    return np.any(active[faces], axis=1)


def evaluate_local_mesh_quality(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    active_vertices: Optional[np.ndarray] = None,
    cfg: Optional[LocalMeshQualityConfig] = None,
) -> dict:
    """Compare a local deformation against its fixed-topology baseline."""
    cfg = cfg or LocalMeshQualityConfig()
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    faces_i = np.asarray(faces, dtype=np.int64)
    finite = bool(np.isfinite(candidate).all())
    same_shape = bool(candidate.shape == baseline.shape)
    if not finite or not same_shape or len(faces_i) == 0:
        return {
            "accepted": False,
            "finite": finite,
            "same_vertex_shape": same_shape,
            "flipped_faces": 0,
            "collapsed_faces": 0,
            "overstretched_edges": 0,
            "max_edge_ratio": float("inf"),
            "min_area_ratio": 0.0,
            "max_normal_angle_deg": 180.0,
            "reason": "non-finite or incompatible candidate",
        }

    face_mask = _active_face_mask(faces_i, active_vertices)
    selected_faces = faces_i[face_mask]
    area0, normal0 = _face_geometry(baseline, selected_faces)
    area1, normal1 = _face_geometry(candidate, selected_faces)
    area_ratio = area1 / np.maximum(area0, cfg.absolute_min_area)
    normal_cos = np.clip(np.sum(normal0 * normal1, axis=1), -1.0, 1.0)
    normal_angle = np.degrees(np.arccos(normal_cos))

    edges = _unique_edges(selected_faces)
    edge0 = np.linalg.norm(baseline[edges[:, 1]] - baseline[edges[:, 0]], axis=1)
    edge1 = np.linalg.norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1)
    edge_ratio = edge1 / np.maximum(edge0, 1e-12)

    flipped = normal_cos <= 0.0
    collapsed = (area1 <= cfg.absolute_min_area) | (area_ratio < cfg.min_area_ratio)
    overstretched = edge_ratio > cfg.max_edge_ratio
    excessive_normal = normal_angle > cfg.max_normal_angle_deg
    accepted = not (
        np.any(flipped)
        or np.any(collapsed)
        or np.any(overstretched)
        or np.any(excessive_normal)
    )
    failures = []
    if np.any(flipped):
        failures.append("flipped faces")
    if np.any(collapsed):
        failures.append("collapsed faces")
    if np.any(overstretched):
        failures.append("edge stretch")
    if np.any(excessive_normal):
        failures.append("normal discontinuity")
    return {
        "accepted": bool(accepted),
        "finite": True,
        "same_vertex_shape": True,
        "checked_faces": int(len(selected_faces)),
        "checked_edges": int(len(edges)),
        "flipped_faces": int(flipped.sum()),
        "collapsed_faces": int(collapsed.sum()),
        "overstretched_edges": int(overstretched.sum()),
        "excessive_normal_faces": int(excessive_normal.sum()),
        "max_edge_ratio": round(float(edge_ratio.max(initial=1.0)), 6),
        "min_area_ratio": round(float(area_ratio.min(initial=1.0)), 6),
        "max_normal_angle_deg": round(float(normal_angle.max(initial=0.0)), 6),
        "reason": "accepted" if accepted else "rejected: " + ", ".join(failures),
    }


def select_valid_deformation_backtrack(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    active_vertices: Optional[np.ndarray] = None,
    cfg: Optional[LocalMeshQualityConfig] = None,
    alphas: tuple[float, ...] = (1.0, 0.75, 0.5, 0.35, 0.25, 0.15, 0.1),
) -> tuple[np.ndarray, dict, float]:
    """Keep the largest deformation scale that passes all hard mesh gates."""
    baseline = np.asarray(baseline_vertices, dtype=np.float32)
    candidate = np.asarray(candidate_vertices, dtype=np.float32)
    delta = candidate - baseline
    attempts = []
    for alpha in alphas:
        scaled = baseline + float(alpha) * delta
        report = evaluate_local_mesh_quality(
            baseline,
            scaled,
            faces,
            active_vertices=active_vertices,
            cfg=cfg,
        )
        attempts.append(
            {
                "alpha": float(alpha),
                "accepted": bool(report["accepted"]),
                "reason": report["reason"],
            }
        )
        if report["accepted"]:
            report["selected_alpha"] = float(alpha)
            report["backtrack_attempts"] = attempts
            return scaled.astype(np.float32), report, float(alpha)
    baseline_report = evaluate_local_mesh_quality(
        baseline,
        baseline,
        faces,
        active_vertices=active_vertices,
        cfg=cfg,
    )
    baseline_report["selected_alpha"] = 0.0
    baseline_report["backtrack_attempts"] = attempts
    baseline_report["accepted"] = False
    baseline_report["reason"] = "rejected all candidates; baseline kept"
    return baseline.copy(), baseline_report, 0.0

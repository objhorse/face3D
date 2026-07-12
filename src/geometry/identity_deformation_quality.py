"""Hard acceptance gates for controlled fixed-topology identity deformation."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class IdentityDeformationQualityConfig:
    max_edge_change_p95: float = 0.05
    max_edge_change: float = 0.12
    min_area_ratio: float = 0.05
    min_baseline_edge_m: float = 0.0001
    min_baseline_area_m2: float = 1e-9


def _edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def evaluate_identity_geometry(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    cfg: Optional[IdentityDeformationQualityConfig] = None,
) -> dict:
    cfg = cfg or IdentityDeformationQualityConfig()
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    candidate = np.asarray(candidate_vertices, dtype=np.float64)
    tri = np.asarray(faces, dtype=np.int64)
    reasons = []
    if baseline.shape != candidate.shape:
        return {"accepted": False, "reason": "vertex_shape_changed"}
    if not np.isfinite(candidate).all():
        return {"accepted": False, "reason": "non_finite_vertices"}
    if not len(tri):
        return {"accepted": False, "reason": "empty_faces"}
    base_tri = baseline[tri]
    cand_tri = candidate[tri]
    base_cross = np.cross(base_tri[:, 1] - base_tri[:, 0], base_tri[:, 2] - base_tri[:, 0])
    cand_cross = np.cross(cand_tri[:, 1] - cand_tri[:, 0], cand_tri[:, 2] - cand_tri[:, 0])
    base_area = 0.5 * np.linalg.norm(base_cross, axis=1)
    cand_area = 0.5 * np.linalg.norm(cand_cross, axis=1)
    normal_dot = np.sum(base_cross * cand_cross, axis=1)
    stable_faces = base_area >= cfg.min_baseline_area_m2
    flipped = (normal_dot <= 0.0) & stable_faces
    area_ratio = cand_area / np.maximum(base_area, 1e-12)
    collapsed = (area_ratio < cfg.min_area_ratio) & stable_faces
    edges = _edges(tri)
    edge0 = np.linalg.norm(baseline[edges[:, 1]] - baseline[edges[:, 0]], axis=1)
    edge1 = np.linalg.norm(candidate[edges[:, 1]] - candidate[edges[:, 0]], axis=1)
    stable_edges = edge0 >= cfg.min_baseline_edge_m
    edge_change = np.abs(edge1[stable_edges] / edge0[stable_edges] - 1.0)
    p95 = float(np.percentile(edge_change, 95))
    maximum = float(edge_change.max(initial=0.0))
    if np.any(flipped):
        reasons.append("flipped_faces")
    if np.any(collapsed):
        reasons.append("collapsed_faces")
    if p95 > cfg.max_edge_change_p95:
        reasons.append("edge_change_p95")
    if maximum > cfg.max_edge_change:
        reasons.append("edge_change_max")
    return {
        "accepted": not reasons,
        "reason": "accepted" if not reasons else ",".join(reasons),
        "flipped_faces": int(flipped.sum()),
        "collapsed_faces": int(collapsed.sum()),
        "edge_change_p95": p95,
        "edge_change_max": maximum,
        "min_area_ratio": float(area_ratio.min(initial=1.0)),
        "ignored_baseline_tiny_edges": int((~stable_edges).sum()),
        "ignored_baseline_tiny_faces": int((~stable_faces).sum()),
    }


def evaluate_attachment_similarity(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    observations: Sequence[object],
    min_improvement_ratio: float = 0.15,
    max_after_median_m: Optional[float] = None,
) -> dict:
    """Require a candidate to improve the trusted attached 3D observations."""
    if not observations:
        return {"accepted": False, "reason": "no_observations"}
    faces_np = np.asarray(faces, dtype=np.int64)
    target = np.asarray([item.target_model_point for item in observations], dtype=np.float64)
    bary = np.asarray([item.bary_coords for item in observations], dtype=np.float64)
    face_ids = np.asarray([item.face_index for item in observations], dtype=np.int64)
    weights = np.asarray([getattr(item, "weight", 1.0) for item in observations], dtype=np.float64)
    base = (np.asarray(baseline_vertices, dtype=np.float64)[faces_np[face_ids]] * bary[:, :, None]).sum(axis=1)
    candidate = (np.asarray(candidate_vertices, dtype=np.float64)[faces_np[face_ids]] * bary[:, :, None]).sum(axis=1)
    before = np.linalg.norm(base - target, axis=1)
    after = np.linalg.norm(candidate - target, axis=1)
    before_median = float(np.median(before))
    after_median = float(np.median(after))
    improvement = 1.0 - after_median / max(before_median, 1e-12)
    accepted = improvement >= min_improvement_ratio
    reason = "accepted" if accepted else "insufficient_attachment_improvement"
    if max_after_median_m is not None and after_median > float(max_after_median_m):
        accepted = False
        reason = "attachment_error_too_large"
    return {
        "accepted": bool(accepted),
        "reason": reason,
        "before_median_m": before_median,
        "after_median_m": after_median,
        "improvement_ratio": float(improvement),
        "weighted_before_mean_m": float(np.average(before, weights=weights)),
        "weighted_after_mean_m": float(np.average(after, weights=weights)),
    }


def select_identity_backtrack(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    faces: np.ndarray,
    similarity_gate: Optional[Callable[[np.ndarray], dict]] = None,
    cfg: Optional[IdentityDeformationQualityConfig] = None,
    alphas: Sequence[float] = (1.0, 0.75, 0.5, 0.35, 0.25),
) -> tuple[np.ndarray, dict, float]:
    baseline = np.asarray(baseline_vertices, dtype=np.float32)
    candidate = np.asarray(candidate_vertices, dtype=np.float32)
    attempts = []
    for alpha in alphas:
        current = baseline + float(alpha) * (candidate - baseline)
        geometry = evaluate_identity_geometry(baseline, current, faces, cfg)
        similarity = {"accepted": True, "reason": "not_checked"}
        if geometry["accepted"] and similarity_gate is not None:
            similarity = similarity_gate(current)
        accepted = bool(geometry["accepted"] and similarity.get("accepted", False))
        attempts.append({"alpha": float(alpha), "accepted": accepted, "geometry": geometry, "similarity": similarity})
        if accepted:
            return current, {"accepted": True, "selected_alpha": float(alpha), "attempts": attempts}, float(alpha)
    return baseline.copy(), {"accepted": False, "selected_alpha": 0.0, "attempts": attempts, "reason": "all_candidates_rejected"}, 0.0

"""Parametric FLAME shape updates driven by trusted profile-depth evidence."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass(frozen=True)
class ProfileShapeFitConfig:
    nose_region_rings: int = 6
    protected_landmark_weight: float = 10.0
    protected_ridge: float = 1e-8
    target_deadband_m: float = 0.0015
    max_coefficient_delta_l2: float = 3.25
    max_coefficient_delta_abs: float = 1.0
    max_protected_p95_displacement_m: float = 0.0015
    max_protected_displacement_m: float = 0.0040


def _barycentric_landmarks(
    values: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    array = np.asarray(values)
    triangles = np.asarray(landmark_triangles, dtype=np.int64)
    weights = np.asarray(barycentric, dtype=np.float64)
    if array.ndim == 2:
        return np.sum(array[triangles] * weights[:, :, None], axis=1)
    if array.ndim == 3:
        return np.sum(array[triangles] * weights[:, :, None, None], axis=1)
    raise ValueError("barycentric values must have shape (V,3) or (V,3,S)")


def _vertex_ring_region(
    faces: np.ndarray,
    seeds: np.ndarray,
    vertex_count: int,
    rings: int,
) -> np.ndarray:
    adjacency = [set() for _index in range(int(vertex_count))]
    for first, second, third in np.asarray(faces, dtype=np.int64).reshape(-1, 3):
        adjacency[int(first)].update((int(second), int(third)))
        adjacency[int(second)].update((int(first), int(third)))
        adjacency[int(third)].update((int(first), int(second)))
    active = set(int(value) for value in np.asarray(seeds).reshape(-1))
    frontier = set(active)
    for _iteration in range(max(0, int(rings))):
        frontier = {
            neighbor
            for vertex in frontier
            for neighbor in adjacency[vertex]
        } - active
        active.update(frontier)
    result = np.zeros(int(vertex_count), dtype=bool)
    result[list(active)] = True
    return result


def _minimum_drift_update(
    metric_row: Any,
    target_delta_m: float,
    protected_rows: Any,
    *,
    ridge: float,
    max_l2: float,
    max_abs: float,
) -> dict[str, Any]:
    row = np.asarray(metric_row, dtype=np.float64).reshape(1, -1)
    protected = np.asarray(protected_rows, dtype=np.float64).reshape(
        -1,
        row.shape[1],
    )
    if not (
        np.isfinite(row).all()
        and np.isfinite(protected).all()
        and np.isfinite(target_delta_m)
    ):
        raise ValueError("profile shape system must be finite")
    if float(np.linalg.norm(row)) <= 1e-12:
        raise ValueError("profile metric has no FLAME shape sensitivity")
    hessian = (
        protected.T @ protected / float(max(1, len(protected)))
        + float(ridge) * np.eye(row.shape[1], dtype=np.float64)
    )
    response = np.linalg.solve(hessian, row.T)
    denominator = float(row @ response)
    if not np.isfinite(denominator) or abs(denominator) <= 1e-15:
        raise ValueError("profile shape system is singular")
    unconstrained = response[:, 0] * (float(target_delta_m) / denominator)
    l2 = float(np.linalg.norm(unconstrained))
    absolute = float(np.max(np.abs(unconstrained), initial=0.0))
    scale = 1.0
    if max_l2 > 0.0 and l2 > float(max_l2):
        scale = min(scale, float(max_l2) / l2)
    if max_abs > 0.0 and absolute > float(max_abs):
        scale = min(scale, float(max_abs) / absolute)
    update = unconstrained * scale
    return {
        "update": update,
        "unconstrained_l2": l2,
        "unconstrained_max_abs": absolute,
        "trust_scale": float(scale),
        "achieved_delta_m": float(row @ update),
    }


def fit_flame_nose_profile_shape(
    *,
    template_vertices: Any,
    shape_basis: Any,
    baseline_shape: Any,
    faces: Any,
    landmark_triangles: Any,
    landmark_barycentric: Any,
    front_rotation: Any,
    front_translation: Any,
    target_nose_to_upper_lip_m: float,
    config: ProfileShapeFitConfig | None = None,
) -> dict[str, Any]:
    """Fit one observed relative-depth metric without free vertex offsets."""
    cfg = config or ProfileShapeFitConfig()
    template = np.asarray(template_vertices, dtype=np.float64).reshape(-1, 3)
    basis = np.asarray(shape_basis, dtype=np.float64)
    if basis.ndim == 2:
        basis = basis.reshape(len(template), 3, -1)
    shape = np.asarray(baseline_shape, dtype=np.float64).reshape(-1)
    if basis.shape != (len(template), 3, len(shape)):
        raise ValueError("FLAME shape basis dimensions do not match parameters")
    triangles = np.asarray(landmark_triangles, dtype=np.int64).reshape(-1, 3)
    barycentric = np.asarray(landmark_barycentric, dtype=np.float64).reshape(
        len(triangles),
        3,
    )
    rotation = np.asarray(front_rotation, dtype=np.float64).reshape(3, 3)
    translation = np.asarray(front_translation, dtype=np.float64).reshape(3)

    baseline_vertices = template + np.einsum("vcs,s->vc", basis, shape)
    landmark_vertices = _barycentric_landmarks(
        baseline_vertices,
        triangles,
        barycentric,
    )
    landmark_basis = _barycentric_landmarks(
        basis,
        triangles,
        barycentric,
    )
    landmark_camera = landmark_vertices @ rotation.T + translation
    landmark_basis_camera = np.einsum(
        "ab,lbs->las",
        rotation,
        landmark_basis,
    )
    baseline_metric = float(
        landmark_camera[51, 2] - landmark_camera[30, 2]
    )
    requested_delta = float(target_nose_to_upper_lip_m) - baseline_metric
    effective_delta = (
        0.0
        if abs(requested_delta) <= float(cfg.target_deadband_m)
        else requested_delta
    )

    nose_region = _vertex_ring_region(
        np.asarray(faces, dtype=np.int64),
        triangles[27:36],
        len(template),
        cfg.nose_region_rings,
    )
    if int(np.count_nonzero(nose_region)) < 16:
        raise ValueError("FLAME nose support region is unexpectedly small")
    basis_camera = np.einsum("ab,vbs->vas", rotation, basis)
    protected_landmark_indices = np.concatenate(
        (np.arange(0, 27), np.arange(36, len(landmark_basis_camera)))
    )
    protected_rows = np.vstack(
        (
            basis_camera[~nose_region].reshape(-1, len(shape)),
            float(cfg.protected_landmark_weight)
            * landmark_basis_camera[protected_landmark_indices].reshape(
                -1,
                len(shape),
            ),
        )
    )
    metric_row = (
        landmark_basis_camera[51, 2] - landmark_basis_camera[30, 2]
    )
    solution = _minimum_drift_update(
        metric_row,
        effective_delta,
        protected_rows,
        ridge=cfg.protected_ridge,
        max_l2=cfg.max_coefficient_delta_l2,
        max_abs=cfg.max_coefficient_delta_abs,
    )
    update = np.asarray(solution["update"], dtype=np.float64)
    candidate_shape = shape + update
    displacement_camera = np.einsum("vcs,s->vc", basis_camera, update)
    displacement = np.linalg.norm(displacement_camera, axis=1)
    protected_displacement = displacement[~nose_region]
    candidate_metric = baseline_metric + float(metric_row @ update)
    issues: list[str] = []
    protected_p95 = float(np.percentile(protected_displacement, 95.0))
    protected_max = float(np.max(protected_displacement, initial=0.0))
    if protected_p95 > float(cfg.max_protected_p95_displacement_m):
        issues.append("protected_p95_displacement_exceeded")
    if protected_max > float(cfg.max_protected_displacement_m):
        issues.append("protected_max_displacement_exceeded")
    warnings: list[str] = []
    if solution["trust_scale"] < 1.0:
        warnings.append("coefficient_trust_region_clipped")

    return {
        "passed": not issues,
        "issues": issues,
        "warnings": warnings,
        "candidate_shape": candidate_shape.astype(np.float32),
        "coefficient_update": update.astype(np.float32),
        "baseline_nose_to_upper_lip_m": baseline_metric,
        "target_nose_to_upper_lip_m": float(target_nose_to_upper_lip_m),
        "requested_delta_m": requested_delta,
        "effective_delta_m": effective_delta,
        "candidate_nose_to_upper_lip_m": candidate_metric,
        "remaining_target_error_m": float(
            target_nose_to_upper_lip_m - candidate_metric
        ),
        "nose_region_vertex_count": int(np.count_nonzero(nose_region)),
        "protected_vertex_count": int(np.count_nonzero(~nose_region)),
        "coefficient_delta_l2": float(np.linalg.norm(update)),
        "coefficient_delta_max_abs": float(
            np.max(np.abs(update), initial=0.0)
        ),
        "trust_scale": float(solution["trust_scale"]),
        "unconstrained_coefficient_delta_l2": float(
            solution["unconstrained_l2"]
        ),
        "unconstrained_coefficient_delta_max_abs": float(
            solution["unconstrained_max_abs"]
        ),
        "displacement_m": {
            "full_mean": float(np.mean(displacement)),
            "full_p95": float(np.percentile(displacement, 95.0)),
            "full_max": float(np.max(displacement, initial=0.0)),
            "nose_mean": float(np.mean(displacement[nose_region])),
            "nose_p95": float(np.percentile(displacement[nose_region], 95.0)),
            "nose_max": float(np.max(displacement[nose_region], initial=0.0)),
            "protected_mean": float(np.mean(protected_displacement)),
            "protected_p95": protected_p95,
            "protected_max": protected_max,
        },
        "config": {
            key: value
            for key, value in vars(cfg).items()
        },
    }


def profile_target_from_silhouette_report(
    silhouette_report: Mapping[str, Any],
) -> float:
    comparison = silhouette_report.get(
        "comparison_to_local_depth_surface",
        {},
    )
    value_mm = comparison.get("nose_tip_minus_upper_lip_mm")
    if value_mm is None or not np.isfinite(float(value_mm)):
        raise ValueError("trusted nose-tip to upper-lip target is missing")
    nose_report = silhouette_report.get("points", {}).get("nose_tip", {})
    if not nose_report.get("passed") or int(nose_report.get("valid_side_count", 0)) < 2:
        raise ValueError("nose profile target lacks two-side silhouette support")
    return float(value_mm) / 1000.0

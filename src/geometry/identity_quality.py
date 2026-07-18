"""MICA-prior drift metrics for FLAME shape optimization.

Drift is measured against MICA initialization to reject runaway deformation.
It is not an independent measurement of whether a candidate resembles the
subject in the source photographs.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, Optional

import numpy as np


@dataclass(frozen=True)
class IdentityDriftThresholds:
    max_coefficient_l2: float = 7.0
    max_mean_displacement_pct: float = 1.5
    max_p95_displacement_pct: float = 2.5
    max_displacement_pct: float = 4.0


def mica_centered_shape_regularization(
    shape_param: Any,
    identity_anchor: Any,
    *,
    anchor_weight: float,
    mean_shape_weight: float,
):
    """Return the differentiable MICA-centered shape regularization loss."""
    return (
        float(anchor_weight) * ((shape_param - identity_anchor) ** 2).mean()
        + float(mean_shape_weight) * (shape_param ** 2).mean()
    )


def compute_identity_drift(
    *,
    anchor_shape: Any,
    candidate_shape: Any,
    anchor_vertices: Any,
    candidate_vertices: Any,
) -> Dict[str, Any]:
    """Measure parameter and neutral-mesh drift from a MICA shape prior."""
    anchor = np.asarray(anchor_shape, dtype=np.float64).reshape(-1)
    candidate = np.asarray(candidate_shape, dtype=np.float64).reshape(-1)
    anchor_mesh = np.asarray(anchor_vertices, dtype=np.float64)
    candidate_mesh = np.asarray(candidate_vertices, dtype=np.float64)

    finite = bool(
        anchor.shape == candidate.shape
        and anchor_mesh.shape == candidate_mesh.shape
        and anchor_mesh.ndim == 2
        and anchor_mesh.shape[-1] == 3
        and np.isfinite(anchor).all()
        and np.isfinite(candidate).all()
        and np.isfinite(anchor_mesh).all()
        and np.isfinite(candidate_mesh).all()
    )
    if not finite:
        return {
            "finite": False,
            "coefficient_delta_l2": float("inf"),
            "coefficient_delta_max": float("inf"),
            "face_width": 0.0,
            "mean_displacement_pct": float("inf"),
            "p95_displacement_pct": float("inf"),
            "max_displacement_pct": float("inf"),
        }

    coefficient_delta = candidate - anchor
    displacement = np.linalg.norm(candidate_mesh - anchor_mesh, axis=1)
    face_width = float(np.ptp(anchor_mesh[:, 0])) if len(anchor_mesh) else 0.0
    if not np.isfinite(face_width) or face_width <= 1e-12:
        return {
            "finite": False,
            "coefficient_delta_l2": float(np.linalg.norm(coefficient_delta)),
            "coefficient_delta_max": float(np.max(np.abs(coefficient_delta), initial=0.0)),
            "face_width": face_width,
            "mean_displacement_pct": float("inf"),
            "p95_displacement_pct": float("inf"),
            "max_displacement_pct": float("inf"),
        }

    displacement_pct = displacement / face_width * 100.0
    return {
        "finite": True,
        "coefficient_delta_l2": float(np.linalg.norm(coefficient_delta)),
        "coefficient_delta_max": float(np.max(np.abs(coefficient_delta), initial=0.0)),
        "face_width": face_width,
        "mean_displacement_pct": float(np.mean(displacement_pct)) if len(displacement_pct) else 0.0,
        "p95_displacement_pct": float(np.percentile(displacement_pct, 95)) if len(displacement_pct) else 0.0,
        "max_displacement_pct": float(np.max(displacement_pct, initial=0.0)),
    }


def make_identity_drift_gate(
    *,
    anchor_shape: Any,
    candidate_shape: Any,
    anchor_vertices: Any,
    candidate_vertices: Any,
    thresholds: Optional[IdentityDriftThresholds] = None,
) -> Dict[str, Any]:
    thresholds = thresholds or IdentityDriftThresholds()
    metrics = compute_identity_drift(
        anchor_shape=anchor_shape,
        candidate_shape=candidate_shape,
        anchor_vertices=anchor_vertices,
        candidate_vertices=candidate_vertices,
    )
    issues = []
    if not metrics["finite"]:
        issues.append("identity_data_not_finite_or_shape_mismatch")
    if metrics["coefficient_delta_l2"] > thresholds.max_coefficient_l2:
        issues.append("coefficient_delta_l2_exceeded")
    if metrics["mean_displacement_pct"] > thresholds.max_mean_displacement_pct:
        issues.append("mean_neutral_mesh_displacement_exceeded")
    if metrics["p95_displacement_pct"] > thresholds.max_p95_displacement_pct:
        issues.append("p95_neutral_mesh_displacement_exceeded")
    if metrics["max_displacement_pct"] > thresholds.max_displacement_pct:
        issues.append("max_neutral_mesh_displacement_exceeded")

    return {
        "passed": not issues,
        "issues": issues,
        "metrics": metrics,
        "thresholds": {
            "max_coefficient_l2": thresholds.max_coefficient_l2,
            "max_mean_displacement_pct": thresholds.max_mean_displacement_pct,
            "max_p95_displacement_pct": thresholds.max_p95_displacement_pct,
            "max_displacement_pct": thresholds.max_displacement_pct,
        },
    }


def select_identity_safe_candidate(
    candidates: Iterable[Dict[str, Any]],
    *,
    observation_tolerance: float = 1.0,
) -> Optional[Dict[str, Any]]:
    """Select only from identity/mesh-safe candidates.

    Observation differences within one source-image pixel are treated as a tie;
    the candidate with less identity drift wins that tie.
    """
    safe = [
        candidate
        for candidate in candidates
        if candidate.get("identity_gate", {}).get("passed")
        and candidate.get("mesh_quality_gate", {}).get("passed")
        and np.isfinite(float(candidate.get("observation_score_px", float("inf"))))
    ]
    if not safe:
        return None

    best_observation = min(float(item["observation_score_px"]) for item in safe)
    tied = [
        item
        for item in safe
        if float(item["observation_score_px"])
        <= best_observation + float(observation_tolerance)
    ]
    return min(
        tied,
        key=lambda item: (
            float(item["identity_gate"]["metrics"]["mean_displacement_pct"]),
            float(item["identity_gate"]["metrics"]["coefficient_delta_l2"]),
            int(item.get("attempt", 0)),
        ),
    )


def select_stable_refinement_checkpoint(
    candidates: Iterable[Dict[str, Any]],
    *,
    observation_tolerance: float,
) -> Optional[Dict[str, Any]]:
    """Choose the earliest safe checkpoint within observation resolution.

    Once contour scores are indistinguishable at render resolution, later
    optimization is unsupported extra deformation rather than extra evidence.
    """
    safe = [
        candidate
        for candidate in candidates
        if candidate.get("identity_gate", {}).get("passed")
        and candidate.get("mesh_quality_gate", {}).get("passed")
        and candidate.get("accepted")
        and np.isfinite(float(candidate.get("observation_score_px", float("inf"))))
    ]
    if not safe:
        return None
    best_observation = min(float(item["observation_score_px"]) for item in safe)
    indistinguishable = [
        item
        for item in safe
        if float(item["observation_score_px"])
        <= best_observation + float(observation_tolerance)
    ]
    return min(
        indistinguishable,
        key=lambda item: (
            int(item.get("step", item.get("attempt", 0))),
            float(item["identity_gate"]["metrics"]["coefficient_delta_l2"]),
        ),
    )

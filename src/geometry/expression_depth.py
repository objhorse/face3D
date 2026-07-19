"""Depth-aware constraints for FLAME expression coefficients."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


@dataclass(frozen=True)
class ExpressionDepthThresholds:
    max_mouth_forward_mean_mm: float = 0.5
    max_mouth_forward_p95_mm: float = 1.5
    max_nose_abs_mean_mm: float = 0.5
    max_chin_abs_mean_mm: float = 0.5
    max_projection_iterations: int = 40
    tolerance_mm: float = 1e-4


def expression_regions_from_landmarks(
    landmark_triangle_vertices: Any,
    n_vertices: int,
) -> dict[str, np.ndarray]:
    triangles = np.asarray(landmark_triangle_vertices, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[0] < 68 or triangles.shape[1] != 3:
        raise ValueError("landmark triangle vertices must have shape (68+, 3)")
    if triangles.min() < 0 or triangles.max() >= int(n_vertices):
        raise ValueError("landmark triangle vertices are outside the FLAME topology")

    def vertices(indices: Sequence[int]) -> np.ndarray:
        return np.unique(triangles[np.asarray(indices, dtype=np.int64)].reshape(-1))

    return {
        "nose": vertices(range(27, 36)),
        "mouth": vertices(range(48, 68)),
        "chin": vertices((7, 8, 9)),
        "eyes": vertices(range(36, 48)),
    }


def _basis_array(expression_basis: Any) -> np.ndarray:
    basis = np.asarray(expression_basis, dtype=np.float64)
    if basis.ndim != 3 or basis.shape[1] != 3:
        raise ValueError("expression basis must have shape (vertices, 3, parameters)")
    if not np.isfinite(basis).all():
        raise ValueError("expression basis contains non-finite values")
    return basis


def _parameters_array(parameters: Any, expected: int) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float64).reshape(-1)
    if len(values) != int(expected):
        raise ValueError(
            f"expression has {len(values)} parameters; expected {int(expected)}"
        )
    if not np.isfinite(values).all():
        raise ValueError("expression parameters contain non-finite values")
    return values


def _region_indices(
    regions: Mapping[str, Sequence[int]],
    name: str,
    n_vertices: int,
) -> np.ndarray:
    values = np.unique(np.asarray(regions.get(name, []), dtype=np.int64).reshape(-1))
    if len(values) == 0:
        raise ValueError(f"expression depth region is empty: {name}")
    if values.min() < 0 or values.max() >= int(n_vertices):
        raise ValueError(f"expression depth region is outside topology: {name}")
    return values


def expression_displacement(
    expression_basis: Any,
    parameters: Any,
) -> np.ndarray:
    basis = _basis_array(expression_basis)
    values = _parameters_array(parameters, basis.shape[2])
    return np.tensordot(basis, values, axes=([2], [0]))


def expression_depth_diagnostics(
    expression_basis: Any,
    parameters: Any,
    regions: Mapping[str, Sequence[int]],
    *,
    thresholds: ExpressionDepthThresholds | None = None,
    forward_axis: int = 2,
    forward_sign: float = 1.0,
    unit_scale: float = 1000.0,
) -> dict[str, Any]:
    limits = thresholds or ExpressionDepthThresholds()
    basis = _basis_array(expression_basis)
    if forward_axis not in (0, 1, 2):
        raise ValueError("forward_axis must be 0, 1, or 2")
    displacement = expression_displacement(basis, parameters)
    region_stats: dict[str, dict[str, float]] = {}
    for name in ("nose", "mouth", "chin", "eyes"):
        indices = _region_indices(regions, name, len(basis))
        forward = (
            displacement[indices, forward_axis]
            * float(forward_sign)
            * float(unit_scale)
        )
        positive = np.maximum(forward, 0.0)
        region_stats[name] = {
            "forward_mean_mm": float(np.mean(forward)),
            "forward_p95_mm": float(np.percentile(positive, 95.0)),
            "absolute_mean_mm": float(np.mean(np.abs(forward))),
            "absolute_p95_mm": float(np.percentile(np.abs(forward), 95.0)),
        }

    issues: list[str] = []
    mouth = region_stats["mouth"]
    if mouth["forward_mean_mm"] > limits.max_mouth_forward_mean_mm:
        issues.append("mouth_forward_mean_exceeded")
    if mouth["forward_p95_mm"] > limits.max_mouth_forward_p95_mm:
        issues.append("mouth_forward_p95_exceeded")
    if region_stats["nose"]["absolute_mean_mm"] > limits.max_nose_abs_mean_mm:
        issues.append("nose_depth_displacement_exceeded")
    if region_stats["chin"]["absolute_mean_mm"] > limits.max_chin_abs_mean_mm:
        issues.append("chin_depth_displacement_exceeded")
    return {
        "passed": not issues,
        "issues": issues,
        "regions": region_stats,
        "thresholds": {
            "max_mouth_forward_mean_mm": float(limits.max_mouth_forward_mean_mm),
            "max_mouth_forward_p95_mm": float(limits.max_mouth_forward_p95_mm),
            "max_nose_abs_mean_mm": float(limits.max_nose_abs_mean_mm),
            "max_chin_abs_mean_mm": float(limits.max_chin_abs_mean_mm),
        },
    }


def _project_halfspace(
    values: np.ndarray,
    influence: np.ndarray,
    upper_bound: float,
    tolerance: float,
) -> tuple[np.ndarray, bool]:
    amount = float(np.dot(influence, values))
    excess = amount - float(upper_bound)
    norm_sq = float(np.dot(influence, influence))
    if excess <= float(tolerance) or norm_sq <= 1e-14:
        return values, False
    return values - (excess / norm_sq) * influence, True


def constrain_expression_depth(
    expression_basis: Any,
    parameters: Any,
    regions: Mapping[str, Sequence[int]],
    *,
    thresholds: ExpressionDepthThresholds | None = None,
    forward_axis: int = 2,
    forward_sign: float = 1.0,
    unit_scale: float = 1000.0,
) -> dict[str, Any]:
    """Project coefficients onto depth limits with minimum local L2 updates."""
    limits = thresholds or ExpressionDepthThresholds()
    basis = _basis_array(expression_basis)
    original = _parameters_array(parameters, basis.shape[2])
    values = original.copy()
    original_report = expression_depth_diagnostics(
        basis,
        original,
        regions,
        thresholds=limits,
        forward_axis=forward_axis,
        forward_sign=forward_sign,
        unit_scale=unit_scale,
    )
    if original_report["passed"]:
        return {
            "changed": False,
            "parameters": original.astype(np.float32),
            "original": original_report,
            "selected": original_report,
            "iterations": 0,
            "coefficient_delta_l2": 0.0,
            "coefficient_delta_max": 0.0,
        }

    mouth = _region_indices(regions, "mouth", len(basis))
    nose = _region_indices(regions, "nose", len(basis))
    chin = _region_indices(regions, "chin", len(basis))
    scale = float(forward_sign) * float(unit_scale)
    mouth_influence = basis[mouth, forward_axis, :] * scale
    constraints: list[tuple[np.ndarray, float]] = [
        (
            np.mean(mouth_influence, axis=0),
            float(limits.max_mouth_forward_mean_mm),
        )
    ]
    # A per-vertex bound is slightly stricter than P95, but avoids a small subset
    # of expression modes creating a sharp protruding lip ridge.
    constraints.extend(
        (row, float(limits.max_mouth_forward_p95_mm))
        for row in mouth_influence
    )
    for _name, indices, bound in (
        ("nose", nose, limits.max_nose_abs_mean_mm),
        ("chin", chin, limits.max_chin_abs_mean_mm),
    ):
        influences = basis[indices, forward_axis, :] * scale
        for influence in influences:
            constraints.append((influence, float(bound)))
            constraints.append((-influence, float(bound)))

    iterations = 0
    for iteration in range(int(limits.max_projection_iterations)):
        changed = False
        for influence, upper_bound in constraints:
            values, projected = _project_halfspace(
                values,
                influence,
                upper_bound,
                float(limits.tolerance_mm),
            )
            changed = changed or projected
        iterations = iteration + 1
        if not changed:
            break

    selected_report = expression_depth_diagnostics(
        basis,
        values,
        regions,
        thresholds=limits,
        forward_axis=forward_axis,
        forward_sign=forward_sign,
        unit_scale=unit_scale,
    )
    if not selected_report["passed"]:
        # Zero expression is a deterministic safety fallback. The caller may
        # reject it on 2D fidelity, but unsafe depth is never returned as safe.
        values = np.zeros_like(original)
        selected_report = expression_depth_diagnostics(
            basis,
            values,
            regions,
            thresholds=limits,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            unit_scale=unit_scale,
        )
        selected_report["fallback"] = "zero_expression"

    delta = values - original
    return {
        "changed": bool(np.any(np.abs(delta) > 1e-8)),
        "parameters": values.astype(np.float32),
        "original": original_report,
        "selected": selected_report,
        "iterations": int(iterations),
        "coefficient_delta_l2": float(np.linalg.norm(delta)),
        "coefficient_delta_max": float(np.max(np.abs(delta))),
    }

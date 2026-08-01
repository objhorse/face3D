"""Low-dimensional Stage C eyelid fitting in eye-local image coordinates."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from src.geometry.eye_state import EYE_NAMES
from src.geometry.eyelid_fit import EYE_GROUPS, project_landmarks
from src.geometry.local_mesh_quality import (
    LocalMeshQualityConfig,
    evaluate_local_mesh_quality,
)
from src.geometry.semantic_eyelid_rig import SemanticEyelidRig


EYELID_PARAMETER_NAMES = (
    "subject_right_vertical",
    "subject_right_aperture",
    "subject_right_bulge",
    "subject_left_vertical",
    "subject_left_aperture",
    "subject_left_bulge",
)


@dataclass(frozen=True)
class SemanticEyelidOptimizationConfig:
    max_vertical_ratio: float = 0.004
    max_aperture_ratio: float = 0.006
    max_bulge_ratio: float = 0.0025
    closed_bulge_prior_ratio: float = 0.0012
    front_view_weight: float = 1.0
    side_view_weight: float = 0.55
    contour_weight: float = 1.0
    gap_weight: float = 0.8
    parameter_prior_weight: float = 0.10
    symmetry_weight: float = 0.04
    deformation_smoothness_weight: float = 0.08
    observation_scale_px: float = 3.0
    max_nfev: int = 240


@dataclass(frozen=True)
class SemanticEyelidOptimizationResult:
    vertices: np.ndarray
    coefficients: np.ndarray
    control_parameters: np.ndarray
    report: dict


def build_view_eye_weights(
    evidence,
    consensus,
) -> dict[str, dict[str, float]]:
    """Turn eye-state evidence into visibility-aware per-view weights."""
    result: dict[str, dict[str, float]] = {}
    for view_name, view_evidence in evidence.items():
        result[view_name] = {}
        for eye_name, state in view_evidence.eyes.items():
            selected = consensus.states[eye_name]
            if state.state == selected and selected in {"open", "closed"}:
                weight = 0.25 + 0.75 * float(state.confidence)
            elif state.state == "uncertain":
                weight = 0.15
            else:
                weight = 0.04
            if view_name == "subject-left" and eye_name == "subject_right":
                weight *= 0.35
            elif view_name == "subject-right" and eye_name == "subject_left":
                weight *= 0.35
            result[view_name][eye_name] = float(weight)
    return result


def control_parameters_from_semantics(
    rig: SemanticEyelidRig,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Map six semantic values to fixed-corner lid control parameters."""
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (len(EYELID_PARAMETER_NAMES),):
        raise ValueError(
            f"semantic eyelid coefficients must have shape "
            f"({len(EYELID_PARAMETER_NAMES)},)"
        )
    by_name = dict(zip(EYELID_PARAMETER_NAMES, values))
    controls = np.zeros((len(rig.control_names), 2), dtype=np.float64)
    for index, control_name in enumerate(rig.control_names):
        eye_name = (
            "subject_right"
            if control_name.startswith("subject_right")
            else "subject_left"
        )
        vertical = by_name[f"{eye_name}_vertical"]
        aperture = by_name[f"{eye_name}_aperture"]
        bulge = by_name[f"{eye_name}_bulge"]
        if "_upper_lid" in control_name:
            controls[index, 0] = vertical + 0.5 * aperture
            controls[index, 1] = bulge
        elif "_lower_lid" in control_name:
            controls[index, 0] = vertical - 0.5 * aperture
            controls[index, 1] = bulge
    return controls


def _control_offsets(
    rig: SemanticEyelidRig,
    control_parameters: np.ndarray,
) -> np.ndarray:
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    return (
        control_parameters[:, 0:1] * vertical[None, :]
        + control_parameters[:, 1:2]
        * np.asarray(rig.control_normals, dtype=np.float64)
    )


def apply_semantic_eyelid_parameters(
    vertices: np.ndarray,
    rig: SemanticEyelidRig,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(vertices, dtype=np.float64)
    controls = control_parameters_from_semantics(rig, coefficients)
    candidate = np.asarray(
        rig.apply(baseline, _control_offsets(rig, controls)),
        dtype=np.float64,
    )
    all_indices = np.arange(len(baseline), dtype=np.int64)
    outside = np.setdiff1d(all_indices, rig.active_vertices)
    candidate[outside] = baseline[outside]
    candidate[rig.protected_vertices] = baseline[rig.protected_vertices]
    return candidate, controls


def _eye_local_coordinates(
    points: np.ndarray,
    corners: tuple[int, int],
) -> tuple[np.ndarray, float]:
    values = np.asarray(points, dtype=np.float64)
    outer = values[int(corners[0]), :2]
    inner = values[int(corners[1]), :2]
    axis = inner - outer
    width = float(np.linalg.norm(axis))
    if not np.isfinite(width) or width <= 1e-6:
        raise ValueError("eye corners do not define a valid local frame")
    x_axis = axis / width
    y_axis = np.array([-x_axis[1], x_axis[0]], dtype=np.float64)
    centered = values[:, :2] - outer[None, :]
    local = np.column_stack(
        (centered @ x_axis / width, centered @ y_axis / width)
    )
    return local, width


def _eye_shape_residual(
    projected: np.ndarray,
    observed: np.ndarray,
    eye_name: str,
) -> tuple[np.ndarray, np.ndarray, float]:
    indices, corners, gap_pairs = EYE_GROUPS[eye_name]
    predicted_local, _predicted_width = _eye_local_coordinates(projected, corners)
    observed_local, observed_width = _eye_local_coordinates(observed, corners)
    interior = np.asarray(indices[1:-1], dtype=np.int64)
    contour = (
        predicted_local[interior] - observed_local[interior]
    ) * observed_width
    gaps = []
    for upper, lower in gap_pairs:
        predicted_gap = predicted_local[upper, 1] - predicted_local[lower, 1]
        observed_gap = observed_local[upper, 1] - observed_local[lower, 1]
        gaps.append((predicted_gap - observed_gap) * observed_width)
    return contour.reshape(-1), np.asarray(gaps, dtype=np.float64), observed_width


def _shape_metrics(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
    view_eye_weights: Mapping[str, Mapping[str, float]],
) -> dict:
    by_view: dict[str, dict[str, float]] = {}
    values = []
    weights = []
    for view_name, camera in cameras.items():
        if view_name not in observed_landmarks:
            continue
        projected = project_landmarks(
            vertices,
            landmark_triangles,
            barycentric,
            camera,
        )
        observed = np.asarray(observed_landmarks[view_name], dtype=np.float64)
        by_view[view_name] = {}
        for eye_name in EYE_NAMES:
            contour, gaps, _width = _eye_shape_residual(
                projected,
                observed,
                eye_name,
            )
            error = float(
                np.sqrt(np.mean(np.concatenate((contour, gaps)) ** 2))
            )
            weight = float(view_eye_weights[view_name][eye_name])
            by_view[view_name][eye_name] = error
            values.append(error * weight)
            weights.append(weight)
    denominator = float(np.sum(weights))
    mean = float(np.sum(values) / denominator) if denominator > 0.0 else float("inf")
    return {"mean_px": mean, "by_view_eye_px": by_view}


def _active_edges(
    faces: np.ndarray,
    active_vertices: np.ndarray,
    *,
    max_edges: int = 2048,
) -> np.ndarray:
    faces_i = np.asarray(faces, dtype=np.int64)
    edges = np.vstack(
        (faces_i[:, [0, 1]], faces_i[:, [1, 2]], faces_i[:, [2, 0]])
    )
    edges.sort(axis=1)
    edges = np.unique(edges, axis=0)
    active_mask = np.zeros(int(faces_i.max()) + 1, dtype=bool)
    active_mask[np.asarray(active_vertices, dtype=np.int64)] = True
    edges = edges[np.any(active_mask[edges], axis=1)]
    if len(edges) > int(max_edges):
        sample = np.linspace(0, len(edges) - 1, int(max_edges), dtype=np.int64)
        edges = edges[sample]
    return edges.astype(np.int64)


def _baseline_result(
    vertices: np.ndarray,
    rig: SemanticEyelidRig,
    *,
    reason: str,
    metrics: dict,
    success: bool = False,
) -> SemanticEyelidOptimizationResult:
    coefficients = np.zeros(len(EYELID_PARAMETER_NAMES), dtype=np.float64)
    controls = control_parameters_from_semantics(rig, coefficients)
    return SemanticEyelidOptimizationResult(
        vertices=np.asarray(vertices, dtype=np.float64).copy(),
        coefficients=coefficients,
        control_parameters=controls,
        report={
            "success": bool(success),
            "reason": reason,
            "parameter_ordering": list(EYELID_PARAMETER_NAMES),
            "before": metrics,
            "after": metrics,
            "outside_support_unchanged": True,
            "protected_corners_unchanged": True,
        },
    )


def fit_semantic_eyelids_stage_c(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    rig: SemanticEyelidRig,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
    eye_states: Mapping[str, str],
    view_eye_weights: Mapping[str, Mapping[str, float]],
    cfg: Optional[SemanticEyelidOptimizationConfig] = None,
    mesh_quality_cfg: Optional[LocalMeshQualityConfig] = None,
) -> SemanticEyelidOptimizationResult:
    """Fit a fixed-corner, six-parameter eyelid model from three views."""
    cfg = cfg or SemanticEyelidOptimizationConfig()
    baseline = np.asarray(vertices, dtype=np.float64)
    faces_i = np.asarray(faces, dtype=np.int64)
    before = _shape_metrics(
        baseline,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        view_eye_weights,
    )
    if any(eye_states.get(name) not in {"open", "closed"} for name in EYE_NAMES):
        return _baseline_result(
            baseline,
            rig,
            reason="uncertain_eye_state",
            metrics=before,
        )
    if not cameras or not set(cameras).intersection(observed_landmarks):
        return _baseline_result(
            baseline,
            rig,
            reason="missing_eye_observations",
            metrics=before,
        )
    if np.isfinite(before["mean_px"]) and before["mean_px"] <= 1e-10:
        return _baseline_result(
            baseline,
            rig,
            reason="already_matched",
            metrics=before,
            success=True,
        )

    face_width = max(float(np.ptp(baseline[:, 0])), 1e-8)
    limits = np.array(
        [
            cfg.max_vertical_ratio,
            cfg.max_aperture_ratio,
            cfg.max_bulge_ratio,
            cfg.max_vertical_ratio,
            cfg.max_aperture_ratio,
            cfg.max_bulge_ratio,
        ],
        dtype=np.float64,
    ) * face_width
    limits = np.maximum(limits, 1e-8)
    initial = np.zeros(len(EYELID_PARAMETER_NAMES), dtype=np.float64)
    for eye_index, eye_name in enumerate(EYE_NAMES):
        if eye_states[eye_name] == "closed":
            initial[eye_index * 3 + 2] = (
                float(cfg.closed_bulge_prior_ratio) * face_width
            )

    edges = _active_edges(faces_i, rig.active_vertices)
    baseline_edge_lengths = np.linalg.norm(
        baseline[edges[:, 1]] - baseline[edges[:, 0]],
        axis=1,
    )
    edge_scale = max(float(np.median(baseline_edge_lengths)), 1e-8)
    observation_scale = max(float(cfg.observation_scale_px), 1e-8)

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        candidate, _controls = apply_semantic_eyelid_parameters(
            baseline,
            rig,
            coefficients,
        )
        parts = []
        for view_name, camera in cameras.items():
            if view_name not in observed_landmarks:
                continue
            projected = project_landmarks(
                candidate,
                landmark_triangles,
                barycentric,
                camera,
            )
            observed = np.asarray(observed_landmarks[view_name], dtype=np.float64)
            view_weight = (
                float(cfg.front_view_weight)
                if view_name == "front"
                else float(cfg.side_view_weight)
            )
            for eye_name in EYE_NAMES:
                evidence_weight = float(view_eye_weights[view_name][eye_name])
                weight = np.sqrt(max(view_weight * evidence_weight, 0.0))
                contour, gaps, _width = _eye_shape_residual(
                    projected,
                    observed,
                    eye_name,
                )
                parts.append(
                    weight
                    * float(cfg.contour_weight)
                    * contour
                    / observation_scale
                )
                parts.append(
                    weight
                    * float(cfg.gap_weight)
                    * gaps
                    / observation_scale
                )
        parts.append(
            float(cfg.parameter_prior_weight) * coefficients / limits
        )
        right = coefficients[:3] / limits[:3]
        left = coefficients[3:] / limits[3:]
        parts.append(float(cfg.symmetry_weight) * (right - left))
        displacement = candidate - baseline
        edge_delta = displacement[edges[:, 1]] - displacement[edges[:, 0]]
        parts.append(
            float(cfg.deformation_smoothness_weight)
            * edge_delta.reshape(-1)
            / edge_scale
        )
        return np.concatenate(parts).astype(np.float64)

    optimization = least_squares(
        residuals,
        initial,
        bounds=(-limits, limits),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(cfg.max_nfev),
    )
    candidate, controls = apply_semantic_eyelid_parameters(
        baseline,
        rig,
        optimization.x,
    )
    quality = evaluate_local_mesh_quality(
        baseline,
        candidate,
        faces_i,
        active_vertices=rig.active_vertices,
        cfg=mesh_quality_cfg,
    )
    all_indices = np.arange(len(baseline), dtype=np.int64)
    outside = np.setdiff1d(all_indices, rig.active_vertices)
    outside_unchanged = bool(
        np.array_equal(candidate[outside], baseline[outside])
    )
    protected_unchanged = bool(
        np.array_equal(
            candidate[rig.protected_vertices],
            baseline[rig.protected_vertices],
        )
    )
    after = _shape_metrics(
        candidate,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        view_eye_weights,
    )
    success = bool(
        optimization.success
        and quality.get("accepted", False)
        and outside_unchanged
        and protected_unchanged
        and np.isfinite(after["mean_px"])
        and after["mean_px"] <= before["mean_px"]
    )
    reason = "success"
    if not optimization.success:
        reason = "optimization_failed"
    elif not quality.get("accepted", False):
        reason = "mesh_quality_failed"
    elif not outside_unchanged:
        reason = "outside_support_changed"
    elif not protected_unchanged:
        reason = "protected_eye_corners_changed"
    elif not np.isfinite(after["mean_px"]) or after["mean_px"] > before["mean_px"]:
        reason = "eye_shape_error_worsened"

    improvement_ratio = (
        float(after["mean_px"] / before["mean_px"])
        if before["mean_px"] > 1e-12
        else 1.0
    )
    report = {
        "success": success,
        "reason": reason,
        "parameter_ordering": list(EYELID_PARAMETER_NAMES),
        "coefficients": optimization.x.astype(float).tolist(),
        "bounds": limits.astype(float).tolist(),
        "before": before,
        "after": after,
        "improvement_ratio": improvement_ratio,
        "outside_support_unchanged": outside_unchanged,
        "protected_corners_unchanged": protected_unchanged,
        "active_vertex_count": int(len(rig.active_vertices)),
        "protected_vertex_count": int(len(rig.protected_vertices)),
        "mesh_quality": quality,
        "optimization": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "nfev": int(optimization.nfev),
            "cost": float(optimization.cost),
            "optimality": float(optimization.optimality),
        },
    }
    return SemanticEyelidOptimizationResult(
        vertices=candidate,
        coefficients=np.asarray(optimization.x, dtype=np.float64),
        control_parameters=controls,
        report=report,
    )

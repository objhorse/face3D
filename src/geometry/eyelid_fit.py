"""Low-dimensional, quality-gated fitting for the semantic eyelid rig."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from src.geometry.local_mesh_quality import (
    LocalMeshQualityConfig,
    evaluate_local_mesh_quality,
    select_valid_deformation_backtrack,
)
from src.geometry.semantic_eyelid_rig import SemanticEyelidRig


EYE_LANDMARKS = np.arange(36, 48, dtype=np.int64)
EYE_GAP_PAIRS = ((37, 41), (38, 40), (43, 47), (44, 46))
EYE_GROUPS = {
    "subject_right": (np.arange(36, 42, dtype=np.int64), (36, 39), ((37, 41), (38, 40))),
    "subject_left": (np.arange(42, 48, dtype=np.int64), (42, 45), ((43, 47), (44, 46))),
}


@dataclass(frozen=True)
class EyelidFitConfig:
    max_control_offset: Optional[float] = None
    bulge_prior: Optional[float] = None
    regularization_weight: float = 0.08
    symmetry_weight: float = 0.06
    gap_weight: float = 0.35
    bulge_weight: float = 0.04
    observation_scale_px: float = 4.0
    side_view_weight: float = 0.75
    max_observation_worsening_px: float = 0.05
    max_nfev: int = 300


@dataclass(frozen=True)
class EyelidFitResult:
    vertices: np.ndarray
    control_offsets: np.ndarray
    parameters: np.ndarray
    report: dict


def project_vertices(vertices: np.ndarray, camera: Mapping[str, object]) -> np.ndarray:
    points = np.asarray(vertices, dtype=np.float64)
    rotation = np.asarray(camera["R"], dtype=np.float64).reshape(3, 3)
    translation = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    intrinsics = np.asarray(camera["K"], dtype=np.float64).reshape(3, 3)
    camera_points = (rotation @ points.T).T + translation[None, :]
    homogeneous = (intrinsics @ camera_points.T).T
    depth = homogeneous[:, 2:3]
    safe_depth = np.where(
        np.abs(depth) < 1e-8,
        np.where(depth < 0.0, -1e-8, 1e-8),
        depth,
    )
    return homogeneous[:, :2] / safe_depth


def project_landmarks(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    camera: Mapping[str, object],
) -> np.ndarray:
    triangles = np.asarray(landmark_triangles, dtype=np.int64)
    bary = np.asarray(barycentric, dtype=np.float64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("landmark_triangles must have shape (landmarks, 3)")
    if bary.shape != triangles.shape:
        raise ValueError("barycentric weights must match landmark_triangles")
    surface_points = (
        np.asarray(vertices, dtype=np.float64)[triangles] * bary[:, :, None]
    ).sum(axis=1)
    return project_vertices(surface_points, camera)


def control_offsets_from_parameters(
    rig: SemanticEyelidRig,
    parameters: np.ndarray,
) -> np.ndarray:
    values = np.asarray(parameters, dtype=np.float32)
    if values.shape != (len(rig.control_names), 2):
        raise ValueError("eyelid parameters must have shape (controls, 2)")
    vertical = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    return (
        values[:, 0:1] * vertical[None, :]
        + values[:, 1:2] * np.asarray(rig.control_normals, dtype=np.float32)
    ).astype(np.float32)


def _eye_reprojection_mean(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
) -> float:
    errors = []
    for view, camera in cameras.items():
        if view not in observed_landmarks:
            continue
        projected = project_landmarks(vertices, landmark_triangles, barycentric, camera)
        observed = np.asarray(observed_landmarks[view], dtype=np.float64)
        if observed.shape[0] < 48 or observed.shape[1] < 2:
            raise ValueError(f"observed landmarks for {view} must have shape (48+, 2+)")
        errors.extend(
            np.linalg.norm(projected[EYE_LANDMARKS] - observed[EYE_LANDMARKS, :2], axis=1)
        )
    return float(np.mean(errors)) if errors else float("inf")


def _eye_shape_error_mean(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
    view_eye_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
) -> float:
    weighted_errors = []
    weights = []
    for view, camera in cameras.items():
        if view not in observed_landmarks:
            continue
        projected = project_landmarks(vertices, landmark_triangles, barycentric, camera)
        observed = np.asarray(observed_landmarks[view], dtype=np.float64)
        for eye_name, (indices, corners, _gap_pairs) in EYE_GROUPS.items():
            predicted_anchor = projected[np.asarray(corners)].mean(axis=0)
            observed_anchor = observed[np.asarray(corners), :2].mean(axis=0)
            delta = (
                projected[indices] - predicted_anchor
                - (observed[indices, :2] - observed_anchor)
            )
            weight = float((view_eye_weights or {}).get(view, {}).get(eye_name, 1.0))
            weighted_errors.extend(np.linalg.norm(delta, axis=1) * weight)
            weights.extend(np.full(len(indices), weight))
    denominator = float(np.sum(weights))
    return float(np.sum(weighted_errors) / denominator) if denominator > 0.0 else float("inf")


def _eye_reprojection_by_view(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
) -> dict[str, float]:
    report = {}
    for view, camera in cameras.items():
        if view not in observed_landmarks:
            continue
        projected = project_landmarks(vertices, landmark_triangles, barycentric, camera)
        observed = np.asarray(observed_landmarks[view], dtype=np.float64)
        report[view] = float(
            np.linalg.norm(
                projected[EYE_LANDMARKS] - observed[EYE_LANDMARKS, :2],
                axis=1,
            ).mean()
        )
    return report


def _initial_parameters(
    rig: SemanticEyelidRig,
    eye_states: Mapping[str, str],
    bulge_prior: float,
) -> np.ndarray:
    initial = np.zeros((len(rig.control_names), 2), dtype=np.float64)
    for index, name in enumerate(rig.control_names):
        eye_name = "subject_right" if "subject_right" in name else "subject_left"
        if eye_states.get(eye_name) == "closed" and (
            "upper_lid" in name or "lower_lid" in name
        ):
            initial[index, 1] = float(bulge_prior)
    return initial


def _symmetry_pairs(rig: SemanticEyelidRig) -> tuple[tuple[int, int], ...]:
    names = {name: index for index, name in enumerate(rig.control_names)}
    requested = (
        ("subject_right_outer_corner", "subject_left_outer_corner"),
        ("subject_right_upper_lid", "subject_left_upper_lid"),
        ("subject_right_inner_corner", "subject_left_inner_corner"),
        ("subject_right_lower_lid", "subject_left_lower_lid"),
    )
    return tuple((names[right], names[left]) for right, left in requested)


def _baseline_result(
    vertices: np.ndarray,
    rig: SemanticEyelidRig,
    *,
    reason: str,
    before_shape_error: float = float("inf"),
    before_absolute_error: float = float("inf"),
) -> EyelidFitResult:
    parameters = np.zeros((len(rig.control_names), 2), dtype=np.float32)
    offsets = control_offsets_from_parameters(rig, parameters)
    return EyelidFitResult(
        vertices=np.asarray(vertices, dtype=np.float32).copy(),
        control_offsets=offsets,
        parameters=parameters,
        report={
            "accepted": False,
            "reason": reason,
            "eye_shape_error_before_px": float(before_shape_error),
            "eye_shape_error_after_px": float(before_shape_error),
            "eye_reprojection_before_px": float(before_absolute_error),
            "eye_reprojection_after_px": float(before_absolute_error),
            "selected_alpha": 0.0,
        },
    )


def fit_semantic_eyelids(
    *,
    vertices: np.ndarray,
    faces: np.ndarray,
    rig: SemanticEyelidRig,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
    eye_states: Mapping[str, str],
    view_eye_weights: Optional[Mapping[str, Mapping[str, float]]] = None,
    cfg: Optional[EyelidFitConfig] = None,
    mesh_quality_cfg: Optional[LocalMeshQualityConfig] = None,
) -> EyelidFitResult:
    """Fit semantic eyelid controls while keeping all other vertices frozen."""
    cfg = cfg or EyelidFitConfig()
    baseline = np.asarray(vertices, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    before_absolute_error = _eye_reprojection_mean(
        baseline,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
    )
    before_shape_error = _eye_shape_error_mean(
        baseline,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        view_eye_weights,
    )
    if any(eye_states.get(name) not in {"open", "closed"} for name in (
        "subject_right",
        "subject_left",
    )):
        return _baseline_result(
            baseline,
            rig,
            reason="uncertain_eye_state",
            before_shape_error=before_shape_error,
            before_absolute_error=before_absolute_error,
        )
    if not cameras or not set(cameras).intersection(observed_landmarks):
        return _baseline_result(
            baseline,
            rig,
            reason="missing_eye_observations",
            before_shape_error=before_shape_error,
            before_absolute_error=before_absolute_error,
        )

    face_width = max(float(np.ptp(baseline[:, 0])), 1e-6)
    max_offset = (
        float(cfg.max_control_offset)
        if cfg.max_control_offset is not None
        else 0.015 * face_width
    )
    bulge_prior = (
        float(cfg.bulge_prior)
        if cfg.bulge_prior is not None
        else 0.003 * face_width
    )
    max_offset = max(max_offset, 1e-8)
    bulge_prior = float(np.clip(bulge_prior, -max_offset, max_offset))
    initial = _initial_parameters(rig, eye_states, bulge_prior)
    symmetry_pairs = _symmetry_pairs(rig)
    observation_scale = max(float(cfg.observation_scale_px), 1e-6)

    def residuals(flat_parameters: np.ndarray) -> np.ndarray:
        parameters = flat_parameters.reshape(len(rig.control_names), 2)
        candidate = rig.apply(
            baseline,
            control_offsets_from_parameters(rig, parameters),
        )
        residual_parts = []
        for view, camera in cameras.items():
            if view not in observed_landmarks:
                continue
            projected = project_landmarks(
                candidate,
                landmark_triangles,
                barycentric,
                camera,
            )
            observed = np.asarray(observed_landmarks[view], dtype=np.float64)
            view_weight = 1.0 if view == "front" else float(cfg.side_view_weight)
            for eye_name, (indices, corners, gap_pairs) in EYE_GROUPS.items():
                evidence_weight = float(
                    (view_eye_weights or {}).get(view, {}).get(eye_name, 1.0)
                )
                weight = view_weight * max(evidence_weight, 0.0)
                predicted_anchor = projected[np.asarray(corners)].mean(axis=0)
                observed_anchor = observed[np.asarray(corners), :2].mean(axis=0)
                point_delta = (
                    projected[indices] - predicted_anchor
                    - (observed[indices, :2] - observed_anchor)
                )
                residual_parts.append(
                    (weight / observation_scale * point_delta).reshape(-1)
                )
                for upper, lower in gap_pairs:
                    predicted_gap = np.linalg.norm(projected[upper] - projected[lower])
                    observed_gap = np.linalg.norm(observed[upper, :2] - observed[lower, :2])
                    residual_parts.append(
                        np.array([
                            weight
                            * float(cfg.gap_weight)
                            * (predicted_gap - observed_gap)
                            / observation_scale
                        ])
                    )
        residual_parts.append(
            float(cfg.regularization_weight) * parameters.reshape(-1) / max_offset
        )
        for right, left in symmetry_pairs:
            residual_parts.append(
                float(cfg.symmetry_weight) * (parameters[right] - parameters[left]) / max_offset
            )
        for index, name in enumerate(rig.control_names):
            eye_name = "subject_right" if "subject_right" in name else "subject_left"
            if "upper_lid" in name or "lower_lid" in name:
                target = bulge_prior if eye_states[eye_name] == "closed" else 0.0
                residual_parts.append(
                    np.array([
                        float(cfg.bulge_weight)
                        * (parameters[index, 1] - target)
                        / max_offset
                    ])
                )
        return np.concatenate(residual_parts).astype(np.float64)

    optimization = least_squares(
        residuals,
        initial.reshape(-1),
        bounds=(-max_offset, max_offset),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(cfg.max_nfev),
    )
    optimized_parameters = optimization.x.reshape(len(rig.control_names), 2)
    optimized_offsets = control_offsets_from_parameters(rig, optimized_parameters)
    raw_candidate = rig.apply(baseline, optimized_offsets)
    raw_absolute_error = _eye_reprojection_mean(
        raw_candidate,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
    )
    raw_shape_error = _eye_shape_error_mean(
        raw_candidate,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        view_eye_weights,
    )
    raw_quality = evaluate_local_mesh_quality(
        baseline,
        raw_candidate,
        faces_i,
        active_vertices=rig.active_vertices,
        cfg=mesh_quality_cfg,
    )
    candidate, mesh_report, selected_alpha = select_valid_deformation_backtrack(
        baseline,
        raw_candidate,
        faces_i,
        active_vertices=rig.active_vertices,
        cfg=mesh_quality_cfg,
    )
    applied_parameters = (float(selected_alpha) * optimized_parameters).astype(np.float32)
    applied_offsets = control_offsets_from_parameters(rig, applied_parameters)
    outside = np.setdiff1d(np.arange(len(baseline)), rig.active_vertices)
    outside_unchanged = bool(np.array_equal(candidate[outside], baseline[outside]))
    after_absolute_error = _eye_reprojection_mean(
        candidate,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
    )
    after_shape_error = _eye_shape_error_mean(
        candidate,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        view_eye_weights,
    )
    observation_accepted = bool(
        np.isfinite(after_shape_error)
        and after_shape_error
        <= before_shape_error + float(cfg.max_observation_worsening_px)
    )
    accepted = bool(
        mesh_report.get("accepted", False)
        and selected_alpha > 0.0
        and outside_unchanged
        and observation_accepted
    )
    reason = "accepted"
    if not mesh_report.get("accepted", False) or selected_alpha <= 0.0:
        reason = "mesh_quality_rejected"
    elif not outside_unchanged:
        reason = "outside_support_changed"
    elif not observation_accepted:
        reason = "eye_observation_worsened"

    if not accepted:
        candidate = baseline.copy()
        applied_parameters = np.zeros_like(applied_parameters)
        applied_offsets = np.zeros_like(applied_offsets)
        after_absolute_error = before_absolute_error
        after_shape_error = before_shape_error
    report = {
        "accepted": accepted,
        "reason": reason,
        "eye_shape_error_before_px": float(before_shape_error),
        "eye_shape_error_after_px": float(after_shape_error),
        "eye_shape_error_delta_px": float(after_shape_error - before_shape_error),
        "eye_reprojection_before_px": float(before_absolute_error),
        "eye_reprojection_after_px": float(after_absolute_error),
        "eye_reprojection_delta_px": float(after_absolute_error - before_absolute_error),
        "selected_alpha": float(selected_alpha if accepted else 0.0),
        "outside_support_unchanged": outside_unchanged,
        "active_vertex_count": int(len(rig.active_vertices)),
        "optimization": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "nfev": int(optimization.nfev),
            "cost": float(optimization.cost),
            "optimality": float(optimization.optimality),
        },
        "diagnostics": {
            "raw_eye_shape_error_px": float(raw_shape_error),
            "raw_eye_reprojection_px": float(raw_absolute_error),
            "baseline_eye_reprojection_by_view_px": _eye_reprojection_by_view(
                baseline,
                landmark_triangles,
                barycentric,
                cameras,
                observed_landmarks,
            ),
            "raw_eye_reprojection_by_view_px": _eye_reprojection_by_view(
                raw_candidate,
                landmark_triangles,
                barycentric,
                cameras,
                observed_landmarks,
            ),
            "raw_parameters": optimized_parameters.astype(float).tolist(),
            "raw_control_offsets": optimized_offsets.astype(float).tolist(),
            "raw_max_control_offset": float(
                np.linalg.norm(optimized_offsets, axis=1).max(initial=0.0)
            ),
            "raw_max_vertex_displacement": float(
                np.linalg.norm(raw_candidate - baseline, axis=1).max(initial=0.0)
            ),
            "raw_mesh_quality": raw_quality,
        },
        "mesh_quality": mesh_report,
    }
    return EyelidFitResult(
        vertices=candidate.astype(np.float32),
        control_offsets=applied_offsets.astype(np.float32),
        parameters=applied_parameters.astype(np.float32),
        report=report,
    )

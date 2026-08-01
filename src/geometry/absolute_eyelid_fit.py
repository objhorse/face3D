"""Absolute eye-to-face alignment metrics and low-dimensional eyelid fitting."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional

import numpy as np
from scipy.optimize import least_squares

from src.geometry.eyelid_fit import project_landmarks
from src.geometry.harmonic_semantic_deformer import HarmonicSemanticDeformer
from src.geometry.local_mesh_quality import (
    LocalMeshQualityConfig,
    evaluate_local_mesh_quality,
    select_valid_deformation_backtrack,
)
from src.geometry.semantic_eyelid_rig import SemanticEyelidRig


FACE_FRAME_INDICES = np.r_[27:31, 48:68].astype(np.int64)
EYE_DEFINITIONS = {
    "subject_right": {
        "indices": np.arange(36, 42, dtype=np.int64),
        "corners": (36, 39),
        "centerline_pairs": ((37, 41), (38, 40)),
    },
    "subject_left": {
        "indices": np.arange(42, 48, dtype=np.int64),
        "corners": (42, 45),
        "centerline_pairs": ((43, 47), (44, 46)),
    },
}
ABSOLUTE_EYELID_PARAMETER_NAMES = tuple(
    f"{eye_name}_{parameter}"
    for eye_name in ("subject_right", "subject_left")
    for parameter in ("center_y", "roll", "width", "aperture", "arch", "bulge")
)


@dataclass(frozen=True)
class AbsoluteEyelidFitConfig:
    max_center_y_ratio: float = 0.025
    max_roll_ratio: float = 0.012
    max_width_ratio: float = 0.025
    max_aperture_ratio: float = 0.015
    max_arch_ratio: float = 0.012
    max_bulge_ratio: float = 0.006
    observation_scale_px: float = 3.0
    side_view_weight: float = 0.55
    center_weight: float = 0.65
    width_weight: float = 0.30
    roll_weight: float = 0.18
    aperture_weight: float = 0.20
    parameter_prior_weight: float = 0.025
    symmetry_weight: float = 0.02
    deformation_smoothness_weight: float = 0.01
    minimum_improvement_ratio: float = 0.01
    side_view_worsening_px: float = 0.35
    max_nfev: int = 320


@dataclass(frozen=True)
class AbsoluteEyelidFitResult:
    vertices: np.ndarray
    coefficients: np.ndarray
    control_offsets: np.ndarray
    report: dict


def fit_face_frame_similarity(
    projected: np.ndarray,
    observed: np.ndarray,
    indices: np.ndarray = FACE_FRAME_INDICES,
) -> np.ndarray:
    """Fit a non-reflecting 2D similarity from stable model anchors to observations."""
    source_all = np.asarray(projected, dtype=np.float64)
    target_all = np.asarray(observed, dtype=np.float64)
    selected = np.asarray(indices, dtype=np.int64)
    if source_all.ndim != 2 or source_all.shape[1] < 2:
        raise ValueError("projected landmarks must have shape (N, 2+)")
    if target_all.shape[0] != source_all.shape[0] or target_all.shape[1] < 2:
        raise ValueError("observed landmarks must match projected landmarks")
    if selected.ndim != 1 or len(selected) < 3:
        raise ValueError("at least three face-frame anchors are required")
    if int(selected.min()) < 0 or int(selected.max()) >= len(source_all):
        raise ValueError("face-frame indices are outside the landmark array")

    source = source_all[selected, :2]
    target = target_all[selected, :2]
    finite = np.isfinite(source).all(axis=1) & np.isfinite(target).all(axis=1)
    source = source[finite]
    target = target[finite]
    if len(source) < 3:
        raise ValueError("at least three finite face-frame anchors are required")

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    source_energy = float(np.sum(source_zero * source_zero))
    if source_energy <= 1e-12:
        raise ValueError("face-frame anchors do not define a valid scale")

    covariance = source_zero.T @ target_zero
    u, singular_values, vt = np.linalg.svd(covariance)
    correction = np.eye(2, dtype=np.float64)
    if np.linalg.det(u @ vt) < 0.0:
        correction[-1, -1] = -1.0
    rotation = u @ correction @ vt
    scale = float(np.sum(singular_values * np.diag(correction)) / source_energy)
    linear = scale * rotation
    translation = target_center - source_center @ linear
    return np.column_stack((linear.T, translation)).astype(np.float64)


def apply_face_frame_similarity(points: np.ndarray, matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64)
    transform = np.asarray(matrix, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("points must have shape (N, 2+)")
    if transform.shape != (2, 3):
        raise ValueError("similarity matrix must have shape (2, 3)")
    return values[:, :2] @ transform[:, :2].T + transform[:, 2]


def _eye_centerline(points: np.ndarray, eye_name: str) -> np.ndarray:
    definition = EYE_DEFINITIONS[eye_name]
    corners = definition["corners"]
    pairs = definition["centerline_pairs"]
    values = np.asarray(points, dtype=np.float64)
    return np.vstack(
        (
            values[int(corners[0]), :2],
            0.5 * (
                values[int(pairs[0][0]), :2]
                + values[int(pairs[0][1]), :2]
            ),
            0.5 * (
                values[int(pairs[1][0]), :2]
                + values[int(pairs[1][1]), :2]
            ),
            values[int(corners[1]), :2],
        )
    )


def _angle_degrees(vector: np.ndarray) -> float:
    value = np.asarray(vector, dtype=np.float64).reshape(2)
    return float(np.degrees(np.arctan2(value[1], value[0])))


def _wrapped_angle_delta(first: float, second: float) -> float:
    return float((first - second + 180.0) % 360.0 - 180.0)


def compute_absolute_eye_metrics(
    projected: np.ndarray,
    observed: np.ndarray,
    *,
    eye_states: Mapping[str, str],
    face_frame_indices: np.ndarray = FACE_FRAME_INDICES,
) -> dict:
    """Measure eye geometry in one shared face frame without per-eye recentering."""
    projected_values = np.asarray(projected, dtype=np.float64)
    observed_values = np.asarray(observed, dtype=np.float64)
    if projected_values.shape[0] < 68 or observed_values.shape[0] < 68:
        raise ValueError("absolute eyelid metrics require at least 68 landmarks")
    matrix = fit_face_frame_similarity(
        projected_values,
        observed_values,
        np.asarray(face_frame_indices, dtype=np.int64),
    )
    aligned = apply_face_frame_similarity(projected_values, matrix)
    anchor_indices = np.asarray(face_frame_indices, dtype=np.int64)
    anchor_error = np.linalg.norm(
        aligned[anchor_indices] - observed_values[anchor_indices, :2],
        axis=1,
    )

    eye_reports = {}
    all_errors = []
    for eye_name, definition in EYE_DEFINITIONS.items():
        state = str(eye_states.get(eye_name, "uncertain"))
        if state == "closed":
            predicted_curve = _eye_centerline(aligned, eye_name)
            observed_curve = _eye_centerline(observed_values, eye_name)
        else:
            indices = np.asarray(definition["indices"], dtype=np.int64)
            predicted_curve = aligned[indices]
            observed_curve = observed_values[indices, :2]
        curve_errors = np.linalg.norm(
            predicted_curve - observed_curve,
            axis=1,
        )
        all_errors.extend(curve_errors.tolist())

        corner_a, corner_b = map(int, definition["corners"])
        predicted_axis = aligned[corner_b] - aligned[corner_a]
        observed_axis = (
            observed_values[corner_b, :2] - observed_values[corner_a, :2]
        )
        predicted_center = 0.5 * (aligned[corner_a] + aligned[corner_b])
        observed_center = 0.5 * (
            observed_values[corner_a, :2] + observed_values[corner_b, :2]
        )
        center_delta = predicted_center - observed_center
        width_delta = float(
            np.linalg.norm(predicted_axis) - np.linalg.norm(observed_axis)
        )
        roll_delta = _wrapped_angle_delta(
            _angle_degrees(predicted_axis),
            _angle_degrees(observed_axis),
        )
        eye_reports[eye_name] = {
            "state": state,
            "mean_px": float(np.mean(curve_errors)),
            "max_px": float(np.max(curve_errors)),
            "center_delta_x_px": float(center_delta[0]),
            "center_delta_y_px": float(center_delta[1]),
            "center_distance_px": float(np.linalg.norm(center_delta)),
            "width_delta_px": width_delta,
            "roll_delta_degrees": roll_delta,
            "curve_errors_px": curve_errors.astype(float).tolist(),
        }

    return {
        "mean_px": float(np.mean(all_errors)) if all_errors else float("inf"),
        "max_px": float(np.max(all_errors)) if all_errors else float("inf"),
        "face_frame_anchor_mean_px": float(np.mean(anchor_error)),
        "face_frame_anchor_max_px": float(np.max(anchor_error)),
        "face_frame_similarity": matrix.astype(float).tolist(),
        "eyes": eye_reports,
    }


def _control_seed_center(
    baseline: np.ndarray,
    rig: SemanticEyelidRig,
    name: str,
) -> np.ndarray:
    indices = np.asarray(rig.control_seeds[name], dtype=np.int64)
    if len(indices) == 0:
        raise ValueError(f"semantic eyelid control has no seeds: {name}")
    return np.asarray(baseline, dtype=np.float64)[indices].mean(axis=0)


def _eye_control_offsets(
    baseline: np.ndarray,
    rig: SemanticEyelidRig,
    eye_name: str,
    coefficients: np.ndarray,
) -> dict[str, np.ndarray]:
    by_name = {
        parameter: float(coefficients[index])
        for index, parameter in enumerate(
            ("center_y", "roll", "width", "aperture", "arch", "bulge")
        )
    }
    semantic_names = {
        "outer": f"{eye_name}_outer_corner",
        "upper": f"{eye_name}_upper_lid",
        "inner": f"{eye_name}_inner_corner",
        "lower": f"{eye_name}_lower_lid",
    }
    outer = _control_seed_center(baseline, rig, semantic_names["outer"])
    inner = _control_seed_center(baseline, rig, semantic_names["inner"])
    horizontal = inner - outer
    horizontal_length = float(np.linalg.norm(horizontal))
    if horizontal_length <= 1e-9:
        raise ValueError(f"{eye_name} corner controls do not define an eye frame")
    horizontal /= horizontal_length
    up = np.array([0.0, 1.0, 0.0], dtype=np.float64)
    up = up - float(np.dot(up, horizontal)) * horizontal
    up_length = float(np.linalg.norm(up))
    if up_length <= 1e-9:
        raise ValueError(f"{eye_name} controls do not define a vertical eye frame")
    up /= up_length
    eye_center = 0.5 * (outer + inner)
    half_width = max(0.5 * horizontal_length, 1e-9)

    normal_indices = {
        name: rig.control_names.index(semantic_name)
        for name, semantic_name in semantic_names.items()
    }
    offsets: dict[str, np.ndarray] = {}
    for role, semantic_name in semantic_names.items():
        seed_center = _control_seed_center(baseline, rig, semantic_name)
        horizontal_coordinate = float(
            np.dot(seed_center - eye_center, horizontal) / half_width
        )
        offset = by_name["center_y"] * up
        offset = offset + by_name["roll"] * horizontal_coordinate * up
        offset = offset + by_name["width"] * horizontal_coordinate * horizontal
        if role == "upper":
            offset = (
                offset
                + 0.5 * by_name["aperture"] * up
                + by_name["arch"] * up
            )
        elif role == "lower":
            offset = (
                offset
                - 0.5 * by_name["aperture"] * up
                + by_name["arch"] * up
            )
        if role in {"upper", "lower"}:
            normal = np.asarray(
                rig.control_normals[normal_indices[role]],
                dtype=np.float64,
            )
            offset = offset + by_name["bulge"] * normal
        offsets[semantic_name] = offset.astype(np.float64)
    return offsets


def absolute_control_offsets_from_parameters(
    baseline: np.ndarray,
    rig: SemanticEyelidRig,
    coefficients: np.ndarray,
) -> np.ndarray:
    values = np.asarray(coefficients, dtype=np.float64).reshape(-1)
    if values.shape != (len(ABSOLUTE_EYELID_PARAMETER_NAMES),):
        raise ValueError(
            "absolute eyelid coefficients must contain exactly "
            f"{len(ABSOLUTE_EYELID_PARAMETER_NAMES)} values"
        )
    offsets_by_name: dict[str, np.ndarray] = {}
    for eye_index, eye_name in enumerate(("subject_right", "subject_left")):
        start = eye_index * 6
        offsets_by_name.update(
            _eye_control_offsets(
                baseline,
                rig,
                eye_name,
                values[start : start + 6],
            )
        )
    return np.asarray(
        [offsets_by_name[name] for name in rig.control_names],
        dtype=np.float64,
    )


def build_absolute_control_offset_basis(
    baseline: np.ndarray,
    rig: SemanticEyelidRig,
) -> np.ndarray:
    basis = np.zeros(
        (
            len(ABSOLUTE_EYELID_PARAMETER_NAMES),
            len(rig.control_names),
            3,
        ),
        dtype=np.float64,
    )
    for parameter_index in range(len(ABSOLUTE_EYELID_PARAMETER_NAMES)):
        coefficients = np.zeros(
            len(ABSOLUTE_EYELID_PARAMETER_NAMES),
            dtype=np.float64,
        )
        coefficients[parameter_index] = 1.0
        basis[parameter_index] = absolute_control_offsets_from_parameters(
            baseline,
            rig,
            coefficients,
        )
    return basis


def apply_absolute_eyelid_parameters(
    vertices: np.ndarray,
    rig: SemanticEyelidRig,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    baseline = np.asarray(vertices, dtype=np.float64)
    offsets = absolute_control_offsets_from_parameters(
        baseline,
        rig,
        coefficients,
    )
    candidate = (
        baseline
        + np.asarray(rig.weights, dtype=np.float64) @ offsets
    )
    outside = np.setdiff1d(
        np.arange(len(baseline), dtype=np.int64),
        np.asarray(rig.active_vertices, dtype=np.int64),
    )
    candidate[outside] = baseline[outside]
    return candidate, offsets


def _parameter_bounds(
    baseline: np.ndarray,
    rig: SemanticEyelidRig,
    cfg: AbsoluteEyelidFitConfig,
) -> np.ndarray:
    active = np.asarray(rig.active_vertices, dtype=np.int64)
    if len(active) == 0:
        raise ValueError("semantic eyelid rig has no active vertices")
    active_width = max(
        float(np.ptp(np.asarray(baseline, dtype=np.float64)[active, 0])),
        1e-6,
    )
    per_eye = np.array(
        [
            cfg.max_center_y_ratio,
            cfg.max_roll_ratio,
            cfg.max_width_ratio,
            cfg.max_aperture_ratio,
            cfg.max_arch_ratio,
            cfg.max_bulge_ratio,
        ],
        dtype=np.float64,
    )
    if np.any(per_eye <= 0.0):
        raise ValueError("absolute eyelid parameter bounds must be positive")
    return np.tile(active_width * per_eye, 2)


def _eye_curve_points(
    points: np.ndarray,
    eye_name: str,
    eye_state: str,
) -> np.ndarray:
    if eye_state == "closed":
        return _eye_centerline(points, eye_name)
    indices = np.asarray(EYE_DEFINITIONS[eye_name]["indices"], dtype=np.int64)
    return np.asarray(points, dtype=np.float64)[indices, :2]


def _view_metrics(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    barycentric: np.ndarray,
    cameras: Mapping[str, Mapping[str, object]],
    observed_landmarks: Mapping[str, np.ndarray],
    eye_states: Mapping[str, str],
    view_eye_weights: Mapping[str, Mapping[str, float]],
) -> dict:
    by_view: dict[str, dict] = {}
    weighted_values = []
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
        metrics = compute_absolute_eye_metrics(
            projected,
            observed_landmarks[view_name],
            eye_states=eye_states,
        )
        by_view[view_name] = metrics
        view_weight = 1.0 if view_name == "front" else 0.55
        for eye_name, eye_metrics in metrics["eyes"].items():
            weight = view_weight * float(
                view_eye_weights.get(view_name, {}).get(eye_name, 1.0)
            )
            weighted_values.append(float(eye_metrics["mean_px"]) * weight)
            weights.append(weight)
    denominator = float(np.sum(weights))
    mean_px = (
        float(np.sum(weighted_values) / denominator)
        if denominator > 0.0
        else float("inf")
    )
    return {"mean_px": mean_px, "by_view": by_view}


def _unique_active_edges(
    faces: np.ndarray,
    active_vertices: np.ndarray,
) -> np.ndarray:
    faces_i = np.asarray(faces, dtype=np.int64)
    active_mask = np.zeros(int(faces_i.max()) + 1, dtype=bool)
    active_mask[np.asarray(active_vertices, dtype=np.int64)] = True
    selected = faces_i[np.any(active_mask[faces_i], axis=1)]
    edges = np.vstack(
        (selected[:, [0, 1]], selected[:, [1, 2]], selected[:, [2, 0]])
    )
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def fit_absolute_semantic_eyelids(
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
    cfg: Optional[AbsoluteEyelidFitConfig] = None,
    mesh_quality_cfg: Optional[LocalMeshQualityConfig] = None,
) -> AbsoluteEyelidFitResult:
    """Fit eye position and shape in one shared face frame across all views."""
    cfg = cfg or AbsoluteEyelidFitConfig()
    source_vertices = np.asarray(vertices)
    output_dtype = (
        source_vertices.dtype
        if np.issubdtype(source_vertices.dtype, np.floating)
        else np.dtype(np.float32)
    )
    baseline = np.asarray(source_vertices, dtype=np.float32)
    faces_i = np.asarray(faces, dtype=np.int64)
    bounds = _parameter_bounds(baseline, rig, cfg)
    before = _view_metrics(
        baseline,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        eye_states,
        view_eye_weights,
    )
    if any(
        eye_states.get(eye_name) not in {"open", "closed"}
        for eye_name in EYE_DEFINITIONS
    ):
        zeros = np.zeros(len(bounds), dtype=np.float32)
        return AbsoluteEyelidFitResult(
            vertices=baseline.copy(),
            coefficients=zeros,
            control_offsets=absolute_control_offsets_from_parameters(
                baseline,
                rig,
                zeros,
            ),
            report={
                "accepted": False,
                "reason": "uncertain_eye_state",
                "before": before,
                "after": before,
            },
        )

    fixed_face_frames: dict[str, np.ndarray] = {}
    for view_name, camera in cameras.items():
        if view_name not in observed_landmarks:
            continue
        baseline_projection = project_landmarks(
            baseline,
            landmark_triangles,
            barycentric,
            camera,
        )
        fixed_face_frames[view_name] = fit_face_frame_similarity(
            baseline_projection,
            observed_landmarks[view_name],
        )
    deformer = HarmonicSemanticDeformer.build(
        baseline,
        faces_i,
        rig,
        build_absolute_control_offset_basis(baseline, rig),
    )

    active_edges = _unique_active_edges(faces_i, rig.active_vertices)
    if len(active_edges) > 3000:
        stride = max(1, len(active_edges) // 3000)
        active_edges = active_edges[::stride]
    baseline_edge_lengths = np.linalg.norm(
        baseline[active_edges[:, 1]] - baseline[active_edges[:, 0]],
        axis=1,
    )
    edge_scale = max(float(np.median(baseline_edge_lengths)), 1e-8)
    observation_scale = max(float(cfg.observation_scale_px), 1e-8)

    def residuals(coefficients: np.ndarray) -> np.ndarray:
        candidate = deformer.apply(coefficients)
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
            aligned = apply_face_frame_similarity(
                projected,
                fixed_face_frames[view_name],
            )
            observed = np.asarray(
                observed_landmarks[view_name],
                dtype=np.float64,
            )
            view_weight = (
                1.0 if view_name == "front" else float(cfg.side_view_weight)
            )
            for eye_name, definition in EYE_DEFINITIONS.items():
                evidence_weight = max(
                    float(
                        view_eye_weights.get(view_name, {}).get(
                            eye_name,
                            1.0,
                        )
                    ),
                    0.0,
                )
                weight = view_weight * evidence_weight
                predicted_curve = _eye_curve_points(
                    aligned,
                    eye_name,
                    eye_states[eye_name],
                )
                observed_curve = _eye_curve_points(
                    observed,
                    eye_name,
                    eye_states[eye_name],
                )
                parts.append(
                    weight
                    * (predicted_curve - observed_curve).reshape(-1)
                    / observation_scale
                )

                corner_a, corner_b = map(int, definition["corners"])
                predicted_center = 0.5 * (
                    aligned[corner_a] + aligned[corner_b]
                )
                observed_center = 0.5 * (
                    observed[corner_a, :2] + observed[corner_b, :2]
                )
                parts.append(
                    weight
                    * float(cfg.center_weight)
                    * (predicted_center - observed_center)
                    / observation_scale
                )
                predicted_axis = aligned[corner_b] - aligned[corner_a]
                observed_axis = (
                    observed[corner_b, :2] - observed[corner_a, :2]
                )
                width_residual = (
                    np.linalg.norm(predicted_axis)
                    - np.linalg.norm(observed_axis)
                )
                parts.append(
                    np.array(
                        [
                            weight
                            * float(cfg.width_weight)
                            * width_residual
                            / observation_scale
                        ]
                    )
                )
                roll_residual = _wrapped_angle_delta(
                    _angle_degrees(predicted_axis),
                    _angle_degrees(observed_axis),
                )
                parts.append(
                    np.array(
                        [
                            weight
                            * float(cfg.roll_weight)
                            * roll_residual
                            / 10.0
                        ]
                    )
                )
                gap_residuals = []
                for upper, lower in definition["centerline_pairs"]:
                    predicted_gap = np.linalg.norm(
                        aligned[int(upper)] - aligned[int(lower)]
                    )
                    observed_gap = np.linalg.norm(
                        observed[int(upper), :2] - observed[int(lower), :2]
                    )
                    gap_residuals.append(predicted_gap - observed_gap)
                parts.append(
                    weight
                    * float(cfg.aperture_weight)
                    * np.asarray(gap_residuals, dtype=np.float64)
                    / observation_scale
                )

        parts.append(
            float(cfg.parameter_prior_weight)
            * np.asarray(coefficients, dtype=np.float64)
            / bounds
        )
        right = np.asarray(coefficients[:6], dtype=np.float64)
        left = np.asarray(coefficients[6:], dtype=np.float64)
        parts.append(
            float(cfg.symmetry_weight)
            * (right - left)
            / (0.5 * (bounds[:6] + bounds[6:]))
        )
        displacement = candidate - baseline
        edge_delta = (
            displacement[active_edges[:, 1]]
            - displacement[active_edges[:, 0]]
        )
        parts.append(
            float(cfg.deformation_smoothness_weight)
            * edge_delta.reshape(-1)
            / edge_scale
        )
        return np.concatenate(parts).astype(np.float64)

    optimization = least_squares(
        residuals,
        np.zeros(len(bounds), dtype=np.float64),
        bounds=(-bounds, bounds),
        loss="soft_l1",
        f_scale=1.0,
        max_nfev=int(cfg.max_nfev),
    )
    raw_coefficients = np.asarray(optimization.x, dtype=np.float64)
    raw_candidate = deformer.apply(raw_coefficients)
    raw_offsets = absolute_control_offsets_from_parameters(
        baseline,
        rig,
        raw_coefficients,
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
    applied_coefficients = (
        float(selected_alpha) * raw_coefficients
    ).astype(np.float32)
    applied_offsets = absolute_control_offsets_from_parameters(
        baseline,
        rig,
        applied_coefficients,
    )
    outside = np.setdiff1d(
        np.arange(len(baseline), dtype=np.int64),
        np.asarray(rig.active_vertices, dtype=np.int64),
    )
    outside_unchanged = bool(np.array_equal(candidate[outside], baseline[outside]))
    after = _view_metrics(
        candidate,
        landmark_triangles,
        barycentric,
        cameras,
        observed_landmarks,
        eye_states,
        view_eye_weights,
    )
    before_mean = float(before["mean_px"])
    after_mean = float(after["mean_px"])
    improvement_ratio = (
        (before_mean - after_mean) / max(before_mean, 1e-8)
        if np.isfinite(before_mean) and np.isfinite(after_mean)
        else float("-inf")
    )
    front_before = before["by_view"].get("front", {}).get("mean_px", before_mean)
    front_after = after["by_view"].get("front", {}).get("mean_px", after_mean)
    side_names = [
        name for name in before["by_view"]
        if name != "front" and name in after["by_view"]
    ]
    side_before = (
        float(np.mean([before["by_view"][name]["mean_px"] for name in side_names]))
        if side_names
        else before_mean
    )
    side_after = (
        float(np.mean([after["by_view"][name]["mean_px"] for name in side_names]))
        if side_names
        else after_mean
    )
    observation_accepted = bool(
        improvement_ratio >= float(cfg.minimum_improvement_ratio)
        and float(front_after) <= float(front_before) + 1e-8
        and side_after
        <= side_before + float(cfg.side_view_worsening_px)
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
        reason = "absolute_observation_not_improved"

    if not accepted:
        candidate = baseline.copy()
        applied_coefficients = np.zeros_like(applied_coefficients)
        applied_offsets = absolute_control_offsets_from_parameters(
            baseline,
            rig,
            applied_coefficients,
        )
        after = before
        selected_alpha = 0.0
    report = {
        "accepted": accepted,
        "reason": reason,
        "parameter_ordering": list(ABSOLUTE_EYELID_PARAMETER_NAMES),
        "coefficients": applied_coefficients.astype(float).tolist(),
        "bounds": bounds.astype(float).tolist(),
        "before": before,
        "after": after,
        "improvement_ratio": float(
            (float(before["mean_px"]) - float(after["mean_px"]))
            / max(float(before["mean_px"]), 1e-8)
        ),
        "selected_alpha": float(selected_alpha),
        "outside_support_unchanged": outside_unchanged,
        "active_vertex_count": int(len(rig.active_vertices)),
        "mesh_quality": mesh_report,
        "optimization": {
            "success": bool(optimization.success),
            "status": int(optimization.status),
            "message": str(optimization.message),
            "nfev": int(optimization.nfev),
            "cost": float(optimization.cost),
            "optimality": float(optimization.optimality),
        },
        "diagnostics": {
            "raw_coefficients": raw_coefficients.astype(float).tolist(),
            "raw_control_offsets": raw_offsets.astype(float).tolist(),
            "raw_mesh_quality": raw_quality,
            "deformer": deformer.diagnostics,
        },
    }
    output_vertices = np.asarray(candidate, dtype=output_dtype)
    output_vertices[outside] = np.asarray(source_vertices, dtype=output_dtype)[outside]
    return AbsoluteEyelidFitResult(
        vertices=output_vertices,
        coefficients=np.asarray(applied_coefficients, dtype=np.float32),
        control_offsets=np.asarray(applied_offsets, dtype=np.float32),
        report=report,
    )

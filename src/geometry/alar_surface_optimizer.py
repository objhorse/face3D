"""Staged robust fitting for the compact outer-alar surface model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.geometry.alar_surface_basis import (
    ALAR_SURFACE_MODE_NAMES,
    AlarSurfaceBasis,
    apply_alar_surface_basis,
)
from src.geometry.alar_surface_observations import (
    AlarCurveTarget,
    AlarSurfaceObservationBundle,
)
from src.geometry.multiview_nasal_objective import (
    CandidateNasalMesh,
    prepare_nasal_projection_context,
    project_multiview_nasal_boundaries_prepared,
)
from src.geometry.nasal_observations import NASAL_VIEWS
from src.geometry.nasal_semantic_basis import NasalSemanticBasis
from src.geometry.observable_flame_subspace import ProjectionView


_VIEW_TRANSLATION_OFFSET = {
    "front": 6,
    "subject-left": 8,
    "subject-right": 10,
}
_STAGE_ACTIVE = {
    "front": (0, 1, 4, 6, 7),
    "profile": (2, 3, 5, 8, 9, 10, 11),
    "joint": tuple(range(12)),
}


def _readonly(value: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _finite_positive(
    name: str,
    value: Real,
    *,
    allow_zero: bool = False,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
        or (float(value) < 0.0 if allow_zero else float(value) <= 0.0)
    ):
        relation = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be finite and {relation}")
    return float(value)


def _bilinear(field: np.ndarray, points: np.ndarray) -> np.ndarray:
    values = np.asarray(field, dtype=np.float64)
    xy = np.asarray(points, dtype=np.float64)
    height, width = values.shape
    x = np.clip(xy[:, 0], 0.0, width - 1.0)
    y = np.clip(xy[:, 1], 0.0, height - 1.0)
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    ax = x - x0
    ay = y - y0
    return (
        (1.0 - ax) * (1.0 - ay) * values[y0, x0]
        + ax * (1.0 - ay) * values[y0, x1]
        + (1.0 - ax) * ay * values[y1, x0]
        + ax * ay * values[y1, x1]
    )


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    triangles = np.asarray(faces, dtype=np.int64)
    return np.unique(
        np.sort(
            np.vstack(
                (
                    triangles[:, [0, 1]],
                    triangles[:, [1, 2]],
                    triangles[:, [2, 0]],
                )
            ),
            axis=1,
        ),
        axis=0,
    )


def _project_vertices(
    vertices: np.ndarray,
    indices: np.ndarray,
    view: ProjectionView,
) -> np.ndarray:
    camera = (
        view.R_model_to_camera @ vertices[indices].T
    ).T + view.t_model_to_camera
    if np.any(camera[:, 2] <= 1e-6):
        raise ValueError(f"{view.name} candidate profile vertices cross camera")
    homogeneous = (view.K @ camera.T).T
    return homogeneous[:, :2] / homogeneous[:, 2:3]


def _prepare_profile_vertex_samples(
    baseline_vertices: np.ndarray,
    projection_basis: NasalSemanticBasis,
    target: AlarCurveTarget,
    view: ProjectionView,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = target.signed_distance_field.shape
    middle_y = int(
        np.clip(
            np.rint(np.median(target.curve_work[:, 1])),
            0,
            height - 1,
        )
    )
    outward_positive_x = (
        target.signed_distance_field[middle_y, width - 1] > 0.0
    )
    vertical_radius = max(
        2.0,
        0.08 * (
            float(np.max(target.curve_work[:, 1]))
            - float(np.min(target.curve_work[:, 1]))
        ),
    )
    candidates_by_label = []
    for label in (
        "subject_left_nose_wing",
        "subject_right_nose_wing",
    ):
        region = np.asarray(projection_basis.region_masks[label], dtype=bool)
        indices = np.flatnonzero(
            region & ~np.asarray(projection_basis.protected_mask, dtype=bool)
        )
        if len(indices) < 8:
            continue
        pixels = _project_vertices(baseline_vertices, indices, view)
        selected = []
        for target_point in target.curve_work:
            vertical = np.abs(pixels[:, 1] - target_point[1])
            local_candidates = np.flatnonzero(vertical <= vertical_radius)
            if len(local_candidates) < 1:
                local_candidates = np.argsort(vertical)[: min(8, len(vertical))]
            x_values = pixels[local_candidates, 0]
            local = (
                int(np.argmax(x_values))
                if outward_positive_x
                else int(np.argmin(x_values))
            )
            selected.append(int(indices[local_candidates[local]]))
        selected = np.asarray(selected, dtype=np.int64)
        keep = np.r_[True, selected[1:] != selected[:-1]]
        selected = selected[keep]
        if len(selected) < 4:
            continue
        selected_pixels = _project_vertices(baseline_vertices, selected, view)
        differences = (
            target.curve_work[:, None, :]
            - selected_pixels[None, :, :]
        )
        score = float(
            np.mean(
                np.sqrt(
                    np.min(
                        np.sum(differences * differences, axis=2),
                        axis=1,
                    )
                )
            )
        )
        candidates_by_label.append((score, label, selected, selected_pixels))
    if not candidates_by_label:
        raise ValueError(f"{target.name} has no usable projected wing region")
    _score, _label, selected, selected_pixels = min(
        candidates_by_label,
        key=lambda value: value[0],
    )
    confidence = _bilinear(target.confidence_field, selected_pixels)
    confidence = np.clip(confidence, 0.0, 1.0)
    if float(np.sum(confidence)) <= 1e-8:
        confidence = np.ones(len(selected), dtype=np.float64)
    provenance = np.column_stack((selected, selected))
    weights = np.column_stack(
        (
            np.ones(len(selected), dtype=np.float64),
            np.zeros(len(selected), dtype=np.float64),
        )
    )
    return provenance, weights, confidence


@dataclass(frozen=True)
class AlarSurfaceOptimizationConfig:
    coefficient_bound: float = 2.5
    nuisance_translation_bound_px: float = 2.0
    coefficient_prior_weight: float = 0.10
    nuisance_prior_weight: float = 0.18
    reverse_coverage_weight: float = 0.45
    smoothness_weight: float = 0.035
    orientation_weight: float = 25.0
    minimum_orientation_ratio: float = 0.55
    robust_f_scale_px: float = 1.5
    max_nfev_per_stage: int = 600
    min_depth: float = 1e-6

    def __post_init__(self) -> None:
        _finite_positive("coefficient_bound", self.coefficient_bound)
        _finite_positive(
            "nuisance_translation_bound_px",
            self.nuisance_translation_bound_px,
        )
        for name in (
            "coefficient_prior_weight",
            "nuisance_prior_weight",
            "reverse_coverage_weight",
            "smoothness_weight",
            "orientation_weight",
        ):
            _finite_positive(name, getattr(self, name), allow_zero=True)
        ratio = _finite_positive(
            "minimum_orientation_ratio",
            self.minimum_orientation_ratio,
        )
        if ratio >= 1.0:
            raise ValueError("minimum_orientation_ratio must be below one")
        _finite_positive("robust_f_scale_px", self.robust_f_scale_px)
        _finite_positive("min_depth", self.min_depth)
        if (
            isinstance(self.max_nfev_per_stage, (bool, np.bool_))
            or not isinstance(self.max_nfev_per_stage, (int, np.integer))
            or int(self.max_nfev_per_stage) <= 0
        ):
            raise ValueError("max_nfev_per_stage must be a positive integer")


@dataclass(frozen=True)
class PreparedAlarTarget:
    target: AlarCurveTarget
    source_vertex_indices: np.ndarray
    source_weights: np.ndarray
    projection_confidence: np.ndarray

    def __post_init__(self) -> None:
        indices = np.asarray(self.source_vertex_indices)
        weights = np.asarray(self.source_weights, dtype=np.float64)
        confidence = np.asarray(self.projection_confidence, dtype=np.float64)
        count = len(indices)
        if indices.shape != (count, 2) or weights.shape != (count, 2):
            raise ValueError("prepared alar provenance must have shape (N, 2)")
        if confidence.shape != (count,) or count < 4:
            raise ValueError("prepared alar target needs at least four samples")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("prepared alar indices must be integers")
        if not np.allclose(weights.sum(axis=1), 1.0, atol=1e-10):
            raise ValueError("prepared alar edge weights must sum to one")
        if not np.isfinite(weights).all() or not np.isfinite(confidence).all():
            raise ValueError("prepared alar target contains non-finite values")
        object.__setattr__(
            self,
            "source_vertex_indices",
            _readonly(indices, np.int64),
        )
        object.__setattr__(self, "source_weights", _readonly(weights, np.float64))
        object.__setattr__(
            self,
            "projection_confidence",
            _readonly(confidence, np.float64),
        )


@dataclass(frozen=True)
class AlarSurfaceOptimizationContext:
    baseline_vertices: np.ndarray
    faces: np.ndarray
    basis: AlarSurfaceBasis
    views: tuple[ProjectionView, ...]
    prepared_targets: tuple[PreparedAlarTarget, ...]
    smoothness_edges: np.ndarray
    orientation_faces: np.ndarray
    orientation_reference_cross: np.ndarray
    orientation_reference_inverse_norm2: np.ndarray

    def __post_init__(self) -> None:
        vertices = np.asarray(self.baseline_vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        self.basis.validate(len(vertices))
        if faces.ndim != 2 or faces.shape[1] != 3:
            raise ValueError("faces must have shape (F, 3)")
        views = tuple(self.views)
        if tuple(view.name for view in views) != NASAL_VIEWS:
            raise ValueError("views must use canonical semantic order")
        targets = tuple(self.prepared_targets)
        if len(targets) != 4:
            raise ValueError("context requires four outer-alar targets")
        object.__setattr__(
            self,
            "baseline_vertices",
            _readonly(vertices, np.float64),
        )
        object.__setattr__(self, "faces", _readonly(faces, np.int64))
        object.__setattr__(self, "views", views)
        object.__setattr__(self, "prepared_targets", targets)
        object.__setattr__(
            self,
            "smoothness_edges",
            _readonly(self.smoothness_edges, np.int64),
        )
        object.__setattr__(
            self,
            "orientation_faces",
            _readonly(self.orientation_faces, np.int64),
        )
        object.__setattr__(
            self,
            "orientation_reference_cross",
            _readonly(self.orientation_reference_cross, np.float64),
        )
        object.__setattr__(
            self,
            "orientation_reference_inverse_norm2",
            _readonly(self.orientation_reference_inverse_norm2, np.float64),
        )

    @property
    def views_by_name(self) -> Mapping[str, ProjectionView]:
        return MappingProxyType({view.name: view for view in self.views})


@dataclass(frozen=True)
class AlarSurfaceStageResult:
    name: str
    success: bool
    parameters: np.ndarray
    active_indices: tuple[int, ...]
    initial_cost: float
    final_cost: float
    nfev: int
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly(self.parameters, np.float64))


@dataclass(frozen=True)
class AlarSurfaceOptimizationResult:
    success: bool
    parameters: np.ndarray
    candidate_vertices: np.ndarray
    stage_results: Mapping[str, AlarSurfaceStageResult]
    baseline_metrics: Mapping[str, object]
    candidate_metrics: Mapping[str, object]
    initial_cost: float
    final_cost: float
    failure_stage: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(self, "parameters", _readonly(self.parameters, np.float64))
        object.__setattr__(
            self,
            "candidate_vertices",
            _readonly(self.candidate_vertices, np.float64),
        )
        object.__setattr__(
            self,
            "stage_results",
            MappingProxyType(dict(self.stage_results)),
        )
        object.__setattr__(
            self,
            "baseline_metrics",
            MappingProxyType(dict(self.baseline_metrics)),
        )
        object.__setattr__(
            self,
            "candidate_metrics",
            MappingProxyType(dict(self.candidate_metrics)),
        )

    @property
    def coefficients(self) -> np.ndarray:
        return self.parameters[:6]

    @property
    def translations_by_view(self) -> Mapping[str, tuple[float, float]]:
        return MappingProxyType(
            {
                view: tuple(
                    float(value)
                    for value in self.parameters[
                        _VIEW_TRANSLATION_OFFSET[view]:
                        _VIEW_TRANSLATION_OFFSET[view] + 2
                    ]
                )
                for view in NASAL_VIEWS
            }
        )

    def to_report(self) -> dict:
        return {
            "success": bool(self.success),
            "failure_stage": self.failure_stage,
            "parameter_ordering": list(ALAR_SURFACE_MODE_NAMES)
            + [
                f"{view}_{axis}_translation_px"
                for view in NASAL_VIEWS
                for axis in ("x", "y")
            ],
            "parameters": [float(value) for value in self.parameters],
            "coefficients": [float(value) for value in self.coefficients],
            "translations_by_view": {
                view: list(values)
                for view, values in self.translations_by_view.items()
            },
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "cost_reduction_ratio": (
                float(self.final_cost / self.initial_cost)
                if self.initial_cost > 0.0
                else 0.0
            ),
            "baseline_metrics": dict(self.baseline_metrics),
            "candidate_metrics": dict(self.candidate_metrics),
            "stages": {
                name: {
                    "success": bool(stage.success),
                    "active_indices": list(stage.active_indices),
                    "initial_cost": float(stage.initial_cost),
                    "final_cost": float(stage.final_cost),
                    "nfev": int(stage.nfev),
                    "message": stage.message,
                }
                for name, stage in self.stage_results.items()
            },
        }


def prepare_alar_surface_optimization_context(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    basis: AlarSurfaceBasis,
    projection_basis: NasalSemanticBasis,
    observations: AlarSurfaceObservationBundle,
    views: Sequence[ProjectionView],
) -> AlarSurfaceOptimizationContext:
    """Freeze baseline silhouette provenance for a smooth six-mode objective."""
    vertices = np.asarray(baseline_vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    basis.validate(len(vertices))
    canonical_views = tuple(views)
    if tuple(view.name for view in canonical_views) != NASAL_VIEWS:
        raise ValueError("views must use canonical semantic order")
    projection_context = prepare_nasal_projection_context(
        topology,
        projection_basis,
        observations.source,
        canonical_views,
    )
    baseline_candidate = CandidateNasalMesh._from_validated(
        vertices,
        projection_context.faces,
        reuse_faces=True,
    )
    projection = project_multiview_nasal_boundaries_prepared(
        baseline_candidate,
        projection_context,
    )
    prepared_targets = []
    for target in observations.targets:
        if target.semantic_view != "front":
            indices, weights, confidence = _prepare_profile_vertex_samples(
                vertices,
                projection_basis,
                target,
                {
                    view.name: view for view in canonical_views
                }[target.semantic_view],
            )
        else:
            projected = projection.by_view[target.semantic_view]
            keep = np.asarray(
                [
                    boundary == target.boundary_name
                    and label in target.source_labels
                    for boundary, label in zip(
                        projected.boundary_names,
                        projected.source_labels,
                    )
                ],
                dtype=bool,
            )
            selected = np.flatnonzero(keep)
            if len(selected) < 4:
                available = {}
                for boundary, label in zip(
                    projected.boundary_names,
                    projected.source_labels,
                ):
                    key = f"{boundary}:{label}"
                    available[key] = available.get(key, 0) + 1
                raise ValueError(
                    f"{target.name} has only {len(selected)} projected alar "
                    f"samples; available={available}"
                )
            if len(selected) > 32:
                selected = selected[
                    np.rint(
                        np.linspace(0, len(selected) - 1, 32)
                    ).astype(np.int64)
                ]
            indices = projected.source_vertex_indices[selected]
            weights = projected.source_weights[selected]
            confidence = projected.confidence[selected]
        prepared_targets.append(
            PreparedAlarTarget(
                target=target,
                source_vertex_indices=indices,
                source_weights=weights,
                projection_confidence=confidence,
            )
        )

    edges = _mesh_edges(topology)
    edge_keep = (
        basis.support_mask[edges[:, 0]]
        | basis.support_mask[edges[:, 1]]
    )
    smoothness_edges = edges[edge_keep]

    active_faces = topology[np.any(basis.support_mask[topology], axis=1)]
    first = vertices[active_faces[:, 1]] - vertices[active_faces[:, 0]]
    second = vertices[active_faces[:, 2]] - vertices[active_faces[:, 0]]
    reference_cross = np.cross(first, second)
    norm2 = np.einsum("ij,ij->i", reference_cross, reference_cross)
    usable = norm2 > max(float(np.median(norm2)) * 1e-8, 1e-18)
    active_faces = active_faces[usable]
    reference_cross = reference_cross[usable]
    inverse_norm2 = 1.0 / norm2[usable]
    if len(active_faces) < 1:
        raise ValueError("outer-alar support has no non-degenerate faces")

    return AlarSurfaceOptimizationContext(
        baseline_vertices=vertices,
        faces=topology,
        basis=basis,
        views=canonical_views,
        prepared_targets=tuple(prepared_targets),
        smoothness_edges=smoothness_edges,
        orientation_faces=active_faces,
        orientation_reference_cross=reference_cross,
        orientation_reference_inverse_norm2=inverse_norm2,
    )


def _project_target(
    candidate_vertices: np.ndarray,
    prepared: PreparedAlarTarget,
    view: ProjectionView,
    translation: np.ndarray,
    min_depth: float,
) -> np.ndarray:
    points = np.sum(
        candidate_vertices[prepared.source_vertex_indices]
        * prepared.source_weights[:, :, None],
        axis=1,
    )
    camera = (view.R_model_to_camera @ points.T).T + view.t_model_to_camera
    if np.any(camera[:, 2] <= float(min_depth)):
        raise ValueError(f"{view.name} outer-alar samples crossed the camera")
    homogeneous = (view.K @ camera.T).T
    pixels = homogeneous[:, :2] / homogeneous[:, 2:3]
    return pixels + np.asarray(translation, dtype=np.float64)


def project_alar_targets(
    parameters: np.ndarray,
    context: AlarSurfaceOptimizationContext,
    config: Optional[AlarSurfaceOptimizationConfig] = None,
) -> Mapping[str, np.ndarray]:
    """Project the fixed semantic alar samples for diagnostics."""
    limits = AlarSurfaceOptimizationConfig() if config is None else config
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("outer-alar parameter vector must have finite shape (12,)")
    candidate = apply_alar_surface_basis(
        context.baseline_vertices,
        context.basis,
        values[:6],
    )
    projected = {}
    for prepared in context.prepared_targets:
        view_name = prepared.target.semantic_view
        offset = _VIEW_TRANSLATION_OFFSET[view_name]
        projected[prepared.target.name] = _readonly(
            _project_target(
                candidate,
                prepared,
                context.views_by_name[view_name],
                values[offset:offset + 2],
                limits.min_depth,
            ),
            np.float64,
        )
    return MappingProxyType(projected)


def _evaluate_data(
    parameters: np.ndarray,
    context: AlarSurfaceOptimizationContext,
    config: AlarSurfaceOptimizationConfig,
) -> tuple[np.ndarray, dict[str, dict[str, float]], np.ndarray]:
    values = np.asarray(parameters, dtype=np.float64)
    if values.shape != (12,) or not np.isfinite(values).all():
        raise ValueError("outer-alar parameter vector must have finite shape (12,)")
    candidate = apply_alar_surface_basis(
        context.baseline_vertices,
        context.basis,
        values[:6],
    )
    residual_parts = []
    metrics = {}
    for prepared in context.prepared_targets:
        target = prepared.target
        offset = _VIEW_TRANSLATION_OFFSET[target.semantic_view]
        pixels = _project_target(
            candidate,
            prepared,
            context.views_by_name[target.semantic_view],
            values[offset:offset + 2],
            config.min_depth,
        )
        signed = _bilinear(target.signed_distance_field, pixels)
        target_confidence = _bilinear(target.confidence_field, pixels)
        confidence = np.sqrt(
            np.clip(
                prepared.projection_confidence * target_confidence,
                0.0,
                1.0,
            )
        )
        if float(np.sum(confidence)) <= 1e-8:
            confidence = np.ones_like(confidence)
        direct = (
            signed
            * confidence
            / np.sqrt(max(float(np.sum(confidence)), 1.0))
        )

        differences = (
            target.curve_work[:, None, :]
            - pixels[None, :, :]
        )
        reverse_distance = np.sqrt(
            np.min(np.sum(differences * differences, axis=2), axis=1)
        )
        reverse = (
            float(config.reverse_coverage_weight)
            * reverse_distance
            / np.sqrt(float(len(reverse_distance)))
        )
        residual_parts.extend((direct, reverse))
        metrics[target.name] = {
            "semantic_view": target.semantic_view,
            "direct_mean_px": float(np.mean(np.abs(signed))),
            "direct_p90_px": float(np.percentile(np.abs(signed), 90)),
            "reverse_mean_px": float(np.mean(reverse_distance)),
            "combined_mean_px": float(
                0.5 * (
                    np.mean(np.abs(signed))
                    + np.mean(reverse_distance)
                )
            ),
            "projected_sample_count": int(len(pixels)),
            "target_sample_count": int(len(target.curve_work)),
        }
    return np.concatenate(residual_parts), metrics, candidate


def _orientation_ratios(
    candidate: np.ndarray,
    context: AlarSurfaceOptimizationContext,
) -> np.ndarray:
    faces = context.orientation_faces
    first = candidate[faces[:, 1]] - candidate[faces[:, 0]]
    second = candidate[faces[:, 2]] - candidate[faces[:, 0]]
    cross = np.cross(first, second)
    return (
        np.einsum(
            "ij,ij->i",
            cross,
            context.orientation_reference_cross,
        )
        * context.orientation_reference_inverse_norm2
    )


def evaluate_alar_surface_residuals(
    parameters: np.ndarray,
    context: AlarSurfaceOptimizationContext,
    config: Optional[AlarSurfaceOptimizationConfig] = None,
) -> tuple[np.ndarray, dict[str, object], np.ndarray]:
    limits = AlarSurfaceOptimizationConfig() if config is None else config
    data, target_metrics, candidate = _evaluate_data(parameters, context, limits)
    values = np.asarray(parameters, dtype=np.float64)
    displacement = candidate - context.baseline_vertices
    edges = context.smoothness_edges
    edge_delta = displacement[edges[:, 0]] - displacement[edges[:, 1]]
    smoothness = (
        float(limits.smoothness_weight)
        * edge_delta.reshape(-1)
        / (
            float(context.basis.unit_scale)
            * np.sqrt(max(len(edges), 1))
        )
    )
    orientation_ratios = _orientation_ratios(candidate, context)
    orientation = (
        float(limits.orientation_weight)
        * np.maximum(
            0.0,
            float(limits.minimum_orientation_ratio) - orientation_ratios,
        )
    )
    prior = float(limits.coefficient_prior_weight) * values[:6]
    nuisance = (
        float(limits.nuisance_prior_weight)
        * values[6:]
        / float(limits.nuisance_translation_bound_px)
    )
    residual = np.concatenate((data, prior, nuisance, smoothness, orientation))
    by_view = {}
    for view in NASAL_VIEWS:
        combined = [
            metric["combined_mean_px"]
            for metric in target_metrics.values()
            if metric["semantic_view"] == view
        ]
        by_view[view] = {
            "combined_mean_px": float(np.mean(combined)),
            "target_count": int(len(combined)),
        }
    metrics = {
        "targets": target_metrics,
        "views": by_view,
        "minimum_orientation_ratio": float(np.min(orientation_ratios)),
    }
    return residual, metrics, candidate


def _cost(residual: np.ndarray) -> float:
    values = np.asarray(residual, dtype=np.float64)
    return 0.5 * float(values @ values)


def fit_multiview_alar_surface(
    context: AlarSurfaceOptimizationContext,
    config: Optional[AlarSurfaceOptimizationConfig] = None,
    initial_parameters: Optional[np.ndarray] = None,
) -> AlarSurfaceOptimizationResult:
    """Fit front, profile, then joint outer-alar evidence."""
    if not isinstance(context, AlarSurfaceOptimizationContext):
        raise ValueError("context must be an AlarSurfaceOptimizationContext")
    limits = AlarSurfaceOptimizationConfig() if config is None else config
    if not isinstance(limits, AlarSurfaceOptimizationConfig):
        raise ValueError("config must be an AlarSurfaceOptimizationConfig")
    parameters = (
        np.zeros(12, dtype=np.float64)
        if initial_parameters is None
        else np.asarray(initial_parameters, dtype=np.float64).copy()
    )
    if parameters.shape != (12,) or not np.isfinite(parameters).all():
        raise ValueError("initial_parameters must have finite shape (12,)")

    baseline_residual, baseline_metrics, _baseline_candidate = (
        evaluate_alar_surface_residuals(parameters, context, limits)
    )
    initial_cost = _cost(baseline_residual)
    coefficient_bound = float(limits.coefficient_bound)
    translation_bound = float(limits.nuisance_translation_bound_px)
    lower = np.r_[
        np.full(6, -coefficient_bound),
        np.full(6, -translation_bound),
    ]
    upper = -lower
    stages = {}
    failure_stage = None
    for stage_name in ("front", "profile", "joint"):
        active = np.asarray(_STAGE_ACTIVE[stage_name], dtype=np.int64)
        stage_initial = parameters.copy()

        def residual(active_values: np.ndarray) -> np.ndarray:
            trial = stage_initial.copy()
            trial[active] = active_values
            evaluated, _metrics, _candidate = evaluate_alar_surface_residuals(
                trial,
                context,
                limits,
            )
            return evaluated

        initial_stage_residual = residual(stage_initial[active])
        solved = least_squares(
            residual,
            stage_initial[active],
            method="trf",
            jac="2-point",
            bounds=(lower[active], upper[active]),
            loss="soft_l1",
            f_scale=float(limits.robust_f_scale_px),
            max_nfev=int(limits.max_nfev_per_stage),
            x_scale="jac",
            ftol=1e-9,
            xtol=1e-9,
            gtol=1e-9,
        )
        parameters[active] = np.asarray(solved.x, dtype=np.float64)
        final_stage_residual = residual(parameters[active])
        success = bool(solved.success) and np.isfinite(final_stage_residual).all()
        stages[stage_name] = AlarSurfaceStageResult(
            name=stage_name,
            success=success,
            parameters=parameters,
            active_indices=tuple(int(value) for value in active),
            initial_cost=_cost(initial_stage_residual),
            final_cost=_cost(final_stage_residual),
            nfev=int(solved.nfev),
            message=str(solved.message),
        )
        if not success:
            failure_stage = stage_name
            break

    final_residual, candidate_metrics, candidate = (
        evaluate_alar_surface_residuals(parameters, context, limits)
    )
    final_cost = _cost(final_residual)
    return AlarSurfaceOptimizationResult(
        success=failure_stage is None and np.isfinite(final_cost),
        parameters=parameters,
        candidate_vertices=candidate,
        stage_results=stages,
        baseline_metrics=baseline_metrics,
        candidate_metrics=candidate_metrics,
        initial_cost=initial_cost,
        final_cost=final_cost,
        failure_stage=failure_stage,
    )


__all__ = [
    "AlarSurfaceOptimizationConfig",
    "AlarSurfaceOptimizationContext",
    "AlarSurfaceOptimizationResult",
    "evaluate_alar_surface_residuals",
    "fit_multiview_alar_surface",
    "prepare_alar_surface_optimization_context",
    "project_alar_targets",
]

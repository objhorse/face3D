"""Staged multi-view optimization of the compact nasal-base semantic model."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares

from src.geometry.nasal_base_observations import (
    NasalBaseObservationBundle,
    NasalBaseViewObservation,
)
from src.geometry.nasal_base_semantic_basis import (
    NASAL_BASE_MODE_NAMES,
    NasalBaseSemanticBasis,
    apply_nasal_base_semantic_basis,
)
from src.geometry.observable_flame_subspace import ProjectionView


NASAL_BASE_OPTIMIZATION_STAGES = ("frontal", "oblique", "joint")
_STAGE_PARAMETER_NAMES = {
    "frontal": (
        "columella_vertical",
        "nostril_width_shared",
        "nostril_width_asymmetry",
        "nostril_height_shared",
        "nostril_height_asymmetry",
    ),
    "oblique": (
        "columella_depth",
        "alar_rim_curvature_shared",
        "alar_rim_curvature_asymmetry",
    ),
    "joint": NASAL_BASE_MODE_NAMES,
}
_STAGE_VIEW_NAMES = {
    "frontal": ("front",),
    "oblique": ("subject-left", "subject-right"),
    "joint": ("front", "subject-left", "subject-right"),
}

__all__ = [
    "NASAL_BASE_OPTIMIZATION_STAGES",
    "NasalBaseOptimizationConfig",
    "NasalBaseOptimizationContext",
    "NasalBaseOptimizationResult",
    "NasalBaseStageResult",
    "fit_staged_nasal_base",
    "project_nasal_base_landmarks",
]


def _readonly_array(value: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _finite_positive(name: str, value: Real, *, allow_zero: bool = False) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite")
    result = float(value)
    if result < 0.0 if allow_zero else result <= 0.0:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class NasalBaseOptimizationConfig:
    coefficient_bound: float = 2.75
    coefficient_prior_weight: float = 2e-3
    robust_f_scale_px: float = 2.0
    max_nfev_per_stage: int = 180
    ftol: float = 1e-10
    xtol: float = 1e-10
    gtol: float = 1e-10
    min_depth: float = 1e-6

    def __post_init__(self) -> None:
        _finite_positive("coefficient_bound", self.coefficient_bound)
        _finite_positive(
            "coefficient_prior_weight",
            self.coefficient_prior_weight,
            allow_zero=True,
        )
        _finite_positive("robust_f_scale_px", self.robust_f_scale_px)
        _finite_positive("ftol", self.ftol)
        _finite_positive("xtol", self.xtol)
        _finite_positive("gtol", self.gtol)
        _finite_positive("min_depth", self.min_depth)
        if (
            isinstance(self.max_nfev_per_stage, (bool, np.bool_))
            or not isinstance(self.max_nfev_per_stage, (int, np.integer))
            or int(self.max_nfev_per_stage) <= 0
        ):
            raise ValueError("max_nfev_per_stage must be a positive integer")


@dataclass(frozen=True)
class NasalBaseOptimizationContext:
    baseline_vertices: np.ndarray
    landmark_triangles: np.ndarray
    landmark_barycentric: np.ndarray
    basis: NasalBaseSemanticBasis
    views: Sequence[ProjectionView]
    observations: NasalBaseObservationBundle

    def __post_init__(self) -> None:
        vertices = np.asarray(self.baseline_vertices, dtype=np.float64)
        triangles = np.asarray(self.landmark_triangles, dtype=np.int64)
        barycentric = np.asarray(self.landmark_barycentric, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("baseline_vertices must have shape (V, 3)")
        if triangles.shape != (68, 3):
            raise ValueError("landmark_triangles must have shape (68, 3)")
        if barycentric.shape != (68, 3):
            raise ValueError("landmark_barycentric must have shape (68, 3)")
        if not np.isfinite(vertices).all() or not np.isfinite(barycentric).all():
            raise ValueError("context geometry must be finite")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("landmark triangles contain invalid vertex indices")
        if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("landmark barycentric coordinates must sum to one")
        if not isinstance(self.basis, NasalBaseSemanticBasis):
            raise ValueError("basis must be a NasalBaseSemanticBasis")
        self.basis.validate(len(vertices))
        views = tuple(self.views)
        if len(views) != 3 or any(not isinstance(view, ProjectionView) for view in views):
            raise ValueError("views must contain three ProjectionView values")
        by_name = {view.name: view for view in views}
        required = {"front", "subject-left", "subject-right"}
        if set(by_name) != required:
            raise ValueError("views must contain exactly the three semantic views")
        if not isinstance(self.observations, NasalBaseObservationBundle):
            raise ValueError("observations must be a NasalBaseObservationBundle")
        if set(self.observations.by_view) != required:
            raise ValueError("observations must contain the three semantic views")
        object.__setattr__(
            self,
            "baseline_vertices",
            _readonly_array(vertices, np.float64),
        )
        object.__setattr__(
            self,
            "landmark_triangles",
            _readonly_array(triangles, np.int64),
        )
        object.__setattr__(
            self,
            "landmark_barycentric",
            _readonly_array(barycentric, np.float64),
        )
        object.__setattr__(
            self,
            "views",
            tuple(by_name[name] for name in ("front", "subject-left", "subject-right")),
        )

    @property
    def views_by_name(self) -> Mapping[str, ProjectionView]:
        return MappingProxyType({view.name: view for view in self.views})


@dataclass(frozen=True)
class NasalBaseStageResult:
    name: str
    success: bool
    coefficients: np.ndarray
    active_parameter_indices: tuple[int, ...]
    active_parameter_names: tuple[str, ...]
    view_names: tuple[str, ...]
    initial_cost: float
    final_cost: float
    nfev: int
    message: str

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            _readonly_array(self.coefficients, np.float64),
        )


@dataclass(frozen=True)
class NasalBaseOptimizationResult:
    success: bool
    coefficients: np.ndarray
    candidate_vertices: np.ndarray
    initial_cost: float
    final_cost: float
    stage_results: Mapping[str, NasalBaseStageResult]
    failure_stage: Optional[str]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "coefficients",
            _readonly_array(self.coefficients, np.float64),
        )
        object.__setattr__(
            self,
            "candidate_vertices",
            _readonly_array(self.candidate_vertices, np.float64),
        )
        object.__setattr__(
            self,
            "stage_results",
            MappingProxyType(dict(self.stage_results)),
        )

    def to_report(self) -> dict:
        return {
            "success": bool(self.success),
            "failure_stage": self.failure_stage,
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "cost_reduction_ratio": (
                float(self.final_cost / self.initial_cost)
                if self.initial_cost > 0.0
                else 0.0
            ),
            "parameter_ordering": list(NASAL_BASE_MODE_NAMES),
            "coefficients": [float(value) for value in self.coefficients],
            "stages": {
                name: {
                    "success": bool(stage.success),
                    "view_names": list(stage.view_names),
                    "active_parameter_names": list(stage.active_parameter_names),
                    "initial_cost": float(stage.initial_cost),
                    "final_cost": float(stage.final_cost),
                    "nfev": int(stage.nfev),
                    "message": stage.message,
                }
                for name, stage in self.stage_results.items()
            },
        }


def project_nasal_base_landmarks(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    landmark_barycentric: np.ndarray,
    view: ProjectionView,
    *,
    min_depth: float = 1e-6,
) -> np.ndarray:
    """Project 68 barycentric landmarks and return nasal indices 31..35."""
    values = np.asarray(vertices, dtype=np.float64)
    triangles = np.asarray(landmark_triangles, dtype=np.int64)
    barycentric = np.asarray(landmark_barycentric, dtype=np.float64)
    landmarks = np.sum(
        values[triangles] * barycentric[:, :, None],
        axis=1,
    )
    camera = (
        view.R_model_to_camera @ landmarks.T
    ).T + view.t_model_to_camera
    if np.any(camera[:, 2] <= float(min_depth)):
        raise ValueError(f"{view.name} projection contains non-positive depth")
    homogeneous = (view.K @ camera.T).T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(projected).all():
        raise ValueError(f"{view.name} projection is non-finite")
    return projected[31:36]


def _data_residuals(
    context: NasalBaseOptimizationContext,
    coefficients: np.ndarray,
    view_names: Sequence[str],
    min_depth: float,
) -> np.ndarray:
    candidate = apply_nasal_base_semantic_basis(
        context.baseline_vertices,
        context.basis,
        coefficients,
    )
    pieces = []
    for view_name in view_names:
        observation: NasalBaseViewObservation = context.observations.by_view[
            view_name
        ]
        projected = project_nasal_base_landmarks(
            candidate,
            context.landmark_triangles,
            context.landmark_barycentric,
            context.views_by_name[view_name],
            min_depth=min_depth,
        )
        local = projected[observation.landmark_68_indices - 31]
        target = np.asarray(observation.target_xy)
        names = observation.anchor_names
        if view_name == "front":
            right_outer = names.index("subject_right_outer")
            left_outer = names.index("subject_left_outer")
            source_vector = local[left_outer] - local[right_outer]
            target_vector = target[left_outer] - target[right_outer]
            source_length = float(np.linalg.norm(source_vector))
            target_length = float(np.linalg.norm(target_vector))
            if source_length <= 1e-8 or target_length <= 1e-8:
                raise ValueError("front outer alar anchors do not define a frame")
            source_angle = float(np.arctan2(source_vector[1], source_vector[0]))
            target_angle = float(np.arctan2(target_vector[1], target_vector[0]))
            angle = target_angle - source_angle
            cosine = np.cos(angle)
            sine = np.sin(angle)
            rotation = np.asarray(((cosine, sine), (-sine, cosine)))
            source_center = 0.5 * (local[right_outer] + local[left_outer])
            target_center = 0.5 * (target[right_outer] + target[left_outer])
            registered = (
                (target_length / source_length)
                * ((local - source_center) @ rotation)
                + target_center
            )
            active = np.asarray(
                tuple(
                    index
                    for index, name in enumerate(names)
                    if name
                    in {
                        "subject_right_inner",
                        "columella",
                        "subject_left_inner",
                    }
                ),
                dtype=np.int64,
            )
        else:
            outer_name = (
                "subject_left_outer"
                if view_name == "subject-left"
                else "subject_right_outer"
            )
            outer = names.index(outer_name)
            registered = local + (target[outer] - local[outer])
            active = np.asarray(
                tuple(index for index in range(len(names)) if index != outer),
                dtype=np.int64,
            )
        # Fixed outer anchors define the local frame and are not shape residuals.
        weight = np.sqrt(
            observation.confidence[active] / max(len(active), 1)
        )
        pieces.append(
            (
                (registered[active] - target[active])
                * weight[:, None]
            ).ravel()
        )
    return np.concatenate(pieces)


def _data_cost(
    context: NasalBaseOptimizationContext,
    coefficients: np.ndarray,
    view_names: Sequence[str],
    min_depth: float,
) -> float:
    residual = _data_residuals(
        context,
        coefficients,
        view_names,
        min_depth,
    )
    return 0.5 * float(residual @ residual)


def _fit_stage(
    context: NasalBaseOptimizationContext,
    stage_name: str,
    initial_coefficients: np.ndarray,
    config: NasalBaseOptimizationConfig,
) -> NasalBaseStageResult:
    name_to_index = {
        name: index for index, name in enumerate(NASAL_BASE_MODE_NAMES)
    }
    active_names = _STAGE_PARAMETER_NAMES[stage_name]
    active = np.asarray(
        tuple(name_to_index[name] for name in active_names),
        dtype=np.int64,
    )
    view_names = _STAGE_VIEW_NAMES[stage_name]
    initial = np.asarray(initial_coefficients, dtype=np.float64).copy()
    initial_cost = _data_cost(
        context,
        initial,
        view_names,
        config.min_depth,
    )

    prior_scale = np.sqrt(float(config.coefficient_prior_weight))

    def residual(active_values: np.ndarray) -> np.ndarray:
        full = initial.copy()
        full[active] = active_values
        data = _data_residuals(
            context,
            full,
            view_names,
            config.min_depth,
        )
        if prior_scale <= 0.0:
            return data
        return np.concatenate((data, prior_scale * full[active]))

    bound = float(config.coefficient_bound)
    solved = least_squares(
        residual,
        initial[active],
        method="trf",
        jac="2-point",
        bounds=(-bound, bound),
        loss="soft_l1",
        f_scale=float(config.robust_f_scale_px),
        ftol=float(config.ftol),
        xtol=float(config.xtol),
        gtol=float(config.gtol),
        max_nfev=int(config.max_nfev_per_stage),
        x_scale="jac",
    )
    coefficients = initial.copy()
    coefficients[active] = np.asarray(solved.x, dtype=np.float64)
    final_cost = _data_cost(
        context,
        coefficients,
        view_names,
        config.min_depth,
    )
    return NasalBaseStageResult(
        name=stage_name,
        success=bool(solved.success) and np.isfinite(final_cost),
        coefficients=coefficients,
        active_parameter_indices=tuple(int(index) for index in active),
        active_parameter_names=tuple(active_names),
        view_names=tuple(view_names),
        initial_cost=initial_cost,
        final_cost=final_cost,
        nfev=int(solved.nfev),
        message=str(solved.message),
    )


def fit_staged_nasal_base(
    context: NasalBaseOptimizationContext,
    config: Optional[NasalBaseOptimizationConfig] = None,
    initial_coefficients: Optional[np.ndarray] = None,
) -> NasalBaseOptimizationResult:
    """Fit frontal, oblique, then balanced joint evidence in fixed order."""
    if not isinstance(context, NasalBaseOptimizationContext):
        raise ValueError("context must be a NasalBaseOptimizationContext")
    cfg = NasalBaseOptimizationConfig() if config is None else config
    if not isinstance(cfg, NasalBaseOptimizationConfig):
        raise ValueError("config must be a NasalBaseOptimizationConfig")
    coefficients = (
        np.zeros(len(NASAL_BASE_MODE_NAMES), dtype=np.float64)
        if initial_coefficients is None
        else np.asarray(initial_coefficients, dtype=np.float64).copy()
    )
    if coefficients.shape != (len(NASAL_BASE_MODE_NAMES),):
        raise ValueError("initial_coefficients must have shape (8,)")
    if not np.isfinite(coefficients).all():
        raise ValueError("initial_coefficients must be finite")
    bound = float(cfg.coefficient_bound)
    if np.any(np.abs(coefficients) > bound):
        raise ValueError("initial coefficients exceed configured bounds")

    all_views = _STAGE_VIEW_NAMES["joint"]
    initial_cost = _data_cost(
        context,
        coefficients,
        all_views,
        cfg.min_depth,
    )
    stage_results = {}
    failure_stage = None
    for stage_name in NASAL_BASE_OPTIMIZATION_STAGES:
        stage = _fit_stage(context, stage_name, coefficients, cfg)
        stage_results[stage_name] = stage
        coefficients = np.asarray(stage.coefficients, dtype=np.float64)
        if not stage.success:
            failure_stage = stage_name
            break

    candidate = apply_nasal_base_semantic_basis(
        context.baseline_vertices,
        context.basis,
        coefficients,
    )
    final_cost = _data_cost(
        context,
        coefficients,
        all_views,
        cfg.min_depth,
    )
    return NasalBaseOptimizationResult(
        success=failure_stage is None,
        coefficients=coefficients,
        candidate_vertices=candidate,
        initial_cost=initial_cost,
        final_cost=final_cost,
        stage_results=stage_results,
        failure_stage=failure_stage,
    )

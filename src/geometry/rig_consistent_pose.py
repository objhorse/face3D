"""Joint head-pose refinement that preserves calibrated rig extrinsics."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from src.geometry.observable_flame_subspace import ProjectionView


RIG_POSE_VIEW_NAMES = ("front", "subject-left", "subject-right")
DEFAULT_POSE_LANDMARK_INDICES = tuple(
    int(index) for index in np.r_[17:31, 36:48]
)

__all__ = [
    "DEFAULT_POSE_LANDMARK_INDICES",
    "RigConsistentPoseConfig",
    "RigConsistentPoseContext",
    "RigConsistentPoseResult",
    "compose_rig_consistent_views",
    "fit_rig_consistent_pose",
    "project_68_landmarks",
]


def _readonly_array(value: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _positive(name: str, value: Real, *, allow_zero: bool = False) -> float:
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
class RigConsistentPoseConfig:
    landmark_indices: tuple[int, ...] = DEFAULT_POSE_LANDMARK_INDICES
    rotation_bound_degrees: float = 8.0
    translation_xy_bound: float = 0.04
    translation_z_bound: float = 0.08
    pose_prior_weight: float = 0.02
    robust_f_scale_px: float = 3.0
    max_nfev: int = 240
    min_depth: float = 1e-6

    def __post_init__(self) -> None:
        indices = tuple(int(index) for index in self.landmark_indices)
        if (
            not indices
            or len(set(indices)) != len(indices)
            or min(indices) < 0
            or max(indices) >= 68
        ):
            raise ValueError("landmark_indices must be unique values in [0, 67]")
        object.__setattr__(self, "landmark_indices", indices)
        _positive("rotation_bound_degrees", self.rotation_bound_degrees)
        _positive("translation_xy_bound", self.translation_xy_bound)
        _positive("translation_z_bound", self.translation_z_bound)
        _positive("pose_prior_weight", self.pose_prior_weight, allow_zero=True)
        _positive("robust_f_scale_px", self.robust_f_scale_px)
        _positive("min_depth", self.min_depth)
        if (
            isinstance(self.max_nfev, (bool, np.bool_))
            or not isinstance(self.max_nfev, (int, np.integer))
            or int(self.max_nfev) <= 0
        ):
            raise ValueError("max_nfev must be a positive integer")


@dataclass(frozen=True)
class RigConsistentPoseContext:
    vertices: np.ndarray
    landmark_triangles: np.ndarray
    landmark_barycentric: np.ndarray
    initial_views: Sequence[ProjectionView]
    observed_landmarks_68: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        triangles = np.asarray(self.landmark_triangles, dtype=np.int64)
        barycentric = np.asarray(self.landmark_barycentric, dtype=np.float64)
        if vertices.ndim != 2 or vertices.shape[1] != 3:
            raise ValueError("vertices must have shape (V, 3)")
        if triangles.shape != (68, 3):
            raise ValueError("landmark_triangles must have shape (68, 3)")
        if barycentric.shape != (68, 3):
            raise ValueError("landmark_barycentric must have shape (68, 3)")
        if not np.isfinite(vertices).all() or not np.isfinite(barycentric).all():
            raise ValueError("geometry must be finite")
        if triangles.min() < 0 or triangles.max() >= len(vertices):
            raise ValueError("landmark triangle indices are invalid")
        if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
            raise ValueError("landmark barycentric coordinates must sum to one")
        views = tuple(self.initial_views)
        if (
            len(views) != 3
            or any(not isinstance(view, ProjectionView) for view in views)
            or tuple(view.name for view in views) != RIG_POSE_VIEW_NAMES
        ):
            raise ValueError("initial_views must use canonical semantic order")
        observed = {}
        if set(self.observed_landmarks_68) != set(RIG_POSE_VIEW_NAMES):
            raise ValueError("observations must contain exactly the three views")
        for view_name in RIG_POSE_VIEW_NAMES:
            points = np.asarray(
                self.observed_landmarks_68[view_name],
                dtype=np.float64,
            )
            if points.shape != (68, 2) or not np.isfinite(points).all():
                raise ValueError(
                    f"{view_name} observed landmarks must be finite (68, 2)"
                )
            observed[view_name] = _readonly_array(points, np.float64)
        object.__setattr__(self, "vertices", _readonly_array(vertices, np.float64))
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
        object.__setattr__(self, "initial_views", views)
        object.__setattr__(
            self,
            "observed_landmarks_68",
            MappingProxyType(observed),
        )


@dataclass(frozen=True)
class RigConsistentPoseResult:
    success: bool
    views: tuple[ProjectionView, ...]
    delta_rotation_vector: np.ndarray
    delta_translation: np.ndarray
    initial_cost: float
    final_cost: float
    nfev: int
    message: str
    saturated_parameters: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "delta_rotation_vector",
            _readonly_array(self.delta_rotation_vector, np.float64),
        )
        object.__setattr__(
            self,
            "delta_translation",
            _readonly_array(self.delta_translation, np.float64),
        )

    def to_report(self) -> dict:
        return {
            "success": bool(self.success),
            "initial_cost": float(self.initial_cost),
            "final_cost": float(self.final_cost),
            "cost_reduction_ratio": (
                float(self.final_cost / self.initial_cost)
                if self.initial_cost > 0.0
                else 0.0
            ),
            "delta_rotation_vector_radians": [
                float(value) for value in self.delta_rotation_vector
            ],
            "delta_rotation_degrees": [
                float(value)
                for value in np.degrees(self.delta_rotation_vector)
            ],
            "delta_translation": [
                float(value) for value in self.delta_translation
            ],
            "saturated_parameters": list(self.saturated_parameters),
            "nfev": int(self.nfev),
            "message": self.message,
        }


def project_68_landmarks(
    vertices: np.ndarray,
    landmark_triangles: np.ndarray,
    landmark_barycentric: np.ndarray,
    view: ProjectionView,
    *,
    min_depth: float = 1e-6,
) -> np.ndarray:
    points = np.sum(
        np.asarray(vertices, dtype=np.float64)[
            np.asarray(landmark_triangles, dtype=np.int64)
        ]
        * np.asarray(landmark_barycentric, dtype=np.float64)[:, :, None],
        axis=1,
    )
    camera = (view.R_model_to_camera @ points.T).T + view.t_model_to_camera
    if np.any(camera[:, 2] <= float(min_depth)):
        raise ValueError(f"{view.name} projection contains non-positive depth")
    homogeneous = (view.K @ camera.T).T
    projected = homogeneous[:, :2] / homogeneous[:, 2:3]
    if not np.isfinite(projected).all():
        raise ValueError(f"{view.name} projection is non-finite")
    return projected


def compose_rig_consistent_views(
    initial_views: Sequence[ProjectionView],
    front_rotation: np.ndarray,
    front_translation: np.ndarray,
) -> tuple[ProjectionView, ...]:
    """Change one shared head pose while preserving every rig transform."""
    views = tuple(initial_views)
    if (
        len(views) != 3
        or tuple(view.name for view in views) != RIG_POSE_VIEW_NAMES
    ):
        raise ValueError("initial_views must use canonical semantic order")
    rotation = np.asarray(front_rotation, dtype=np.float64)
    translation = np.asarray(front_translation, dtype=np.float64)
    if rotation.shape != (3, 3) or translation.shape != (3,):
        raise ValueError("front pose must be a 3x3 rotation and 3-vector")
    front = views[0]
    result = []
    for view in views:
        relative_rotation = (
            view.R_model_to_camera @ front.R_model_to_camera.T
        )
        relative_translation = (
            view.t_model_to_camera
            - relative_rotation @ front.t_model_to_camera
        )
        result.append(
            ProjectionView(
                name=view.name,
                K=view.K,
                R_model_to_camera=relative_rotation @ rotation,
                t_model_to_camera=(
                    relative_rotation @ translation + relative_translation
                ),
            )
        )
    return tuple(result)


def _data_residuals(
    context: RigConsistentPoseContext,
    views: Sequence[ProjectionView],
    indices: np.ndarray,
    min_depth: float,
) -> np.ndarray:
    pieces = []
    view_weight = 1.0 / np.sqrt(float(len(indices)))
    for view in views:
        projected = project_68_landmarks(
            context.vertices,
            context.landmark_triangles,
            context.landmark_barycentric,
            view,
            min_depth=min_depth,
        )
        observed = context.observed_landmarks_68[view.name]
        pieces.append(
            ((projected[indices] - observed[indices]) * view_weight).ravel()
        )
    return np.concatenate(pieces)


def fit_rig_consistent_pose(
    context: RigConsistentPoseContext,
    config: Optional[RigConsistentPoseConfig] = None,
) -> RigConsistentPoseResult:
    """Fit one shared model-to-rig pose without changing relative cameras."""
    if not isinstance(context, RigConsistentPoseContext):
        raise ValueError("context must be a RigConsistentPoseContext")
    cfg = RigConsistentPoseConfig() if config is None else config
    if not isinstance(cfg, RigConsistentPoseConfig):
        raise ValueError("config must be a RigConsistentPoseConfig")
    indices = np.asarray(cfg.landmark_indices, dtype=np.int64)
    front = context.initial_views[0]
    zero = np.zeros(6, dtype=np.float64)
    rotation_bound = np.deg2rad(float(cfg.rotation_bound_degrees))
    bounds = np.asarray(
        (
            rotation_bound,
            rotation_bound,
            rotation_bound,
            float(cfg.translation_xy_bound),
            float(cfg.translation_xy_bound),
            float(cfg.translation_z_bound),
        ),
        dtype=np.float64,
    )
    prior_scale = np.sqrt(float(cfg.pose_prior_weight))

    def views_for(parameters: np.ndarray) -> tuple[ProjectionView, ...]:
        delta_rotation = Rotation.from_rotvec(parameters[:3]).as_matrix()
        return compose_rig_consistent_views(
            context.initial_views,
            delta_rotation @ front.R_model_to_camera,
            front.t_model_to_camera + parameters[3:],
        )

    def data(parameters: np.ndarray) -> np.ndarray:
        return _data_residuals(
            context,
            views_for(parameters),
            indices,
            cfg.min_depth,
        )

    def residual(parameters: np.ndarray) -> np.ndarray:
        values = data(parameters)
        if prior_scale <= 0.0:
            return values
        return np.concatenate((values, prior_scale * parameters / bounds))

    initial_residual = data(zero)
    initial_cost = 0.5 * float(initial_residual @ initial_residual)
    solved = least_squares(
        residual,
        zero,
        jac="2-point",
        method="trf",
        bounds=(-bounds, bounds),
        loss="soft_l1",
        f_scale=float(cfg.robust_f_scale_px),
        x_scale="jac",
        max_nfev=int(cfg.max_nfev),
        ftol=1e-11,
        xtol=1e-11,
        gtol=1e-11,
    )
    parameters = np.asarray(solved.x, dtype=np.float64)
    corrected_views = views_for(parameters)
    final_residual = _data_residuals(
        context,
        corrected_views,
        indices,
        cfg.min_depth,
    )
    final_cost = 0.5 * float(final_residual @ final_residual)
    parameter_names = ("rx", "ry", "rz", "tx", "ty", "tz")
    saturated = tuple(
        name
        for name, value, limit in zip(parameter_names, parameters, bounds)
        if abs(float(value)) >= float(limit) - 1e-7
    )
    return RigConsistentPoseResult(
        success=bool(solved.success) and np.isfinite(final_cost),
        views=corrected_views,
        delta_rotation_vector=parameters[:3],
        delta_translation=parameters[3:],
        initial_cost=initial_cost,
        final_cost=final_cost,
        nfev=int(solved.nfev),
        message=str(solved.message),
        saturated_parameters=saturated,
    )

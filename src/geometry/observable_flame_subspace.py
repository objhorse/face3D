"""Deterministic observable nasal subspaces of the FLAME shape basis."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Sequence

import numpy as np


OBSERVABLE_FLAME_VIEW_NAMES = ("front", "subject-left", "subject-right")

__all__ = [
    "OBSERVABLE_FLAME_VIEW_NAMES",
    "ObservableFlameSubspaceConfig",
    "ObservableFlameSubspaceResult",
    "ProjectionView",
    "ProjectionViewMetadata",
    "build_observable_flame_subspace",
    "perspective_projection_jacobian",
]

_ROTATION_TOLERANCE = 1e-5
_INTRINSIC_TOLERANCE = 1e-10


def _readonly_array(value: np.ndarray, dtype=None) -> np.ndarray:
    """Snapshot an array into an immutable bytes-backed NumPy view."""
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _freeze_report(value):
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze_report(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze_report(item) for item in value)
    if isinstance(value, np.generic):
        return value.item()
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    raise ValueError("report data must contain only JSON-serializable values")


def _thaw_report(value):
    if isinstance(value, Mapping):
        return {str(key): _thaw_report(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_report(item) for item in value]
    return value


def _finite_real(name: str, value, *, minimum=None, maximum=None) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{name} must be at least {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{name} must be at most {maximum}")
    return result


def _validate_rotation(value: np.ndarray) -> np.ndarray:
    try:
        rotation = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("model-to-camera rotation must be numeric") from exc
    if rotation.shape != (3, 3):
        raise ValueError("model-to-camera rotation must have shape (3, 3)")
    if not np.isfinite(rotation).all():
        raise ValueError("model-to-camera rotation must be finite")
    rigidity_error = float(np.max(np.abs(rotation @ rotation.T - np.eye(3))))
    determinant = float(np.linalg.det(rotation))
    if rigidity_error > _ROTATION_TOLERANCE:
        raise ValueError("model-to-camera rotation must be rigid")
    if determinant <= 0.0 or abs(determinant - 1.0) > _ROTATION_TOLERANCE:
        raise ValueError("model-to-camera rotation must have determinant +1")
    left, _singular_values, right_t = np.linalg.svd(rotation)
    projected = left @ right_t
    if float(np.linalg.det(projected)) <= 0.0:
        raise ValueError("model-to-camera rotation must not be a reflection")
    return projected


def _validate_intrinsics(value: np.ndarray) -> np.ndarray:
    try:
        intrinsic = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("camera intrinsic K must be numeric") from exc
    if intrinsic.shape != (3, 3):
        raise ValueError("camera intrinsic K must have shape (3, 3)")
    if not np.isfinite(intrinsic).all():
        raise ValueError("camera intrinsic K must be finite")
    if not np.allclose(
        intrinsic[2],
        np.array([0.0, 0.0, 1.0]),
        atol=_INTRINSIC_TOLERANCE,
        rtol=0.0,
    ):
        raise ValueError("camera intrinsic K must have final row [0, 0, 1]")
    if intrinsic[0, 0] <= 0.0 or intrinsic[1, 1] <= 0.0:
        raise ValueError("camera intrinsic focal lengths must be positive")
    if abs(float(np.linalg.det(intrinsic))) <= _INTRINSIC_TOLERANCE:
        raise ValueError("camera intrinsic K must be nonsingular")
    return intrinsic


@dataclass(frozen=True)
class ProjectionView:
    """Frozen direct model-to-camera pinhole projection for one semantic view.

    ``subject-left`` and ``subject-right`` name the visible side of the subject,
    never the image side or an operator-relative camera label.
    """

    name: str
    K: np.ndarray
    R_model_to_camera: np.ndarray
    t_model_to_camera: np.ndarray

    def __post_init__(self) -> None:
        name = str(self.name)
        if name not in OBSERVABLE_FLAME_VIEW_NAMES:
            raise ValueError(
                "view name must be one of front, subject-left, subject-right"
            )
        intrinsic = _readonly_array(_validate_intrinsics(self.K), np.float64)
        rotation = _readonly_array(
            _validate_rotation(self.R_model_to_camera),
            np.float64,
        )
        try:
            translation_value = np.asarray(
                self.t_model_to_camera,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("model-to-camera translation must be numeric") from exc
        if translation_value.shape != (3,):
            raise ValueError("model-to-camera translation must have shape (3,)")
        if not np.isfinite(translation_value).all():
            raise ValueError("model-to-camera translation must be finite")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "K", intrinsic)
        object.__setattr__(self, "R_model_to_camera", rotation)
        object.__setattr__(
            self,
            "t_model_to_camera",
            _readonly_array(translation_value, np.float64),
        )


@dataclass(frozen=True)
class ObservableFlameSubspaceConfig:
    """Fixed subject-independent screening and observability thresholds."""

    min_nasal_response_ratio: float = 0.55
    max_protected_to_nasal_energy_ratio: float = 0.75
    relative_nasal_energy_floor: float = 0.03
    relative_singular_value_threshold: float = 1e-3
    max_rank: int = 24
    min_depth: float = 1e-6
    screening_epsilon: float = 1e-12
    singular_value_epsilon: float = 1e-12

    def __post_init__(self) -> None:
        _finite_real(
            "min_nasal_response_ratio",
            self.min_nasal_response_ratio,
            minimum=0.0,
            maximum=1.0,
        )
        _finite_real(
            "max_protected_to_nasal_energy_ratio",
            self.max_protected_to_nasal_energy_ratio,
            minimum=0.0,
        )
        _finite_real(
            "relative_nasal_energy_floor",
            self.relative_nasal_energy_floor,
            minimum=0.0,
            maximum=1.0,
        )
        relative_singular = _finite_real(
            "relative_singular_value_threshold",
            self.relative_singular_value_threshold,
            minimum=0.0,
            maximum=1.0,
        )
        if relative_singular <= 0.0:
            raise ValueError(
                "relative_singular_value_threshold must be greater than zero"
            )
        if (
            isinstance(self.max_rank, (bool, np.bool_))
            or not isinstance(self.max_rank, (int, np.integer))
            or not 1 <= int(self.max_rank) <= 300
        ):
            raise ValueError("max_rank must be an integer in [1, 300]")
        for name in ("min_depth", "screening_epsilon", "singular_value_epsilon"):
            if _finite_real(name, getattr(self, name), minimum=0.0) <= 0.0:
                raise ValueError(f"{name} must be greater than zero")


@dataclass(frozen=True)
class ProjectionViewMetadata:
    """Intrinsic normalization, row scaling, and rank for one view block."""

    name: str
    row_count: int
    support_vertex_count: int
    intrinsic_normalization_matrix: tuple[
        tuple[float, float],
        tuple[float, float],
    ]
    row_scale: float
    pixel_frobenius_norm: float
    balanced_frobenius_norm: float
    rank: int

    def __post_init__(self) -> None:
        if self.name not in OBSERVABLE_FLAME_VIEW_NAMES:
            raise ValueError("view metadata name is invalid")
        for name in ("row_count", "support_vertex_count", "rank"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 0
            ):
                raise ValueError(f"{name} must be a non-negative integer")
        try:
            normalization = np.asarray(
                self.intrinsic_normalization_matrix,
                dtype=np.float64,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "intrinsic_normalization_matrix must be numeric"
            ) from exc
        if normalization.shape != (2, 2) or not np.isfinite(
            normalization
        ).all():
            raise ValueError(
                "intrinsic_normalization_matrix must be a finite 2x2 matrix"
            )
        normalization_tuple = tuple(
            tuple(float(item) for item in row)
            for row in normalization
        )
        object.__setattr__(
            self,
            "intrinsic_normalization_matrix",
            normalization_tuple,
        )
        if _finite_real("row_scale", self.row_scale, minimum=0.0) <= 0.0:
            raise ValueError("row_scale must be greater than zero")
        for name in ("pixel_frobenius_norm", "balanced_frobenius_norm"):
            _finite_real(name, getattr(self, name), minimum=0.0)
        if int(self.row_count) != 2 * int(self.support_vertex_count):
            raise ValueError(
                "row_count must equal twice the support_vertex_count"
            )
        if int(self.rank) > int(self.row_count):
            raise ValueError("view rank must not exceed row_count")


@dataclass(frozen=True)
class ObservableFlameSubspaceResult:
    """Immutable screened and observable mapping into original FLAME space."""

    coefficient_basis: np.ndarray
    vertex_basis: np.ndarray
    candidate_mode_indices: np.ndarray
    screening_pass_mask: np.ndarray
    nasal_mean_squared_energy: np.ndarray
    outside_mean_squared_energy: np.ndarray
    protected_mean_squared_energy: np.ndarray
    nasal_response_ratio: np.ndarray
    outside_to_nasal_energy_ratio: np.ndarray
    protected_to_nasal_energy_ratio: np.ndarray
    relative_nasal_energy: np.ndarray
    singular_values: np.ndarray
    retained_rank: int
    view_names: tuple[str, ...]
    per_view_metadata: tuple[ProjectionViewMetadata, ...]
    config: ObservableFlameSubspaceConfig
    report_data: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.config, ObservableFlameSubspaceConfig):
            raise ValueError("config must be an ObservableFlameSubspaceConfig")
        if (
            isinstance(self.retained_rank, (bool, np.bool_))
            or not isinstance(self.retained_rank, (int, np.integer))
            or int(self.retained_rank) < 0
        ):
            raise ValueError("retained_rank must be a non-negative integer")

        coefficient_basis = _readonly_array(self.coefficient_basis, np.float64)
        vertex_basis = _readonly_array(self.vertex_basis, np.float64)
        candidate_indices = _readonly_array(
            self.candidate_mode_indices,
            np.int64,
        )
        pass_mask = _readonly_array(self.screening_pass_mask, bool)
        metric_names = (
            "nasal_mean_squared_energy",
            "outside_mean_squared_energy",
            "protected_mean_squared_energy",
            "nasal_response_ratio",
            "outside_to_nasal_energy_ratio",
            "protected_to_nasal_energy_ratio",
            "relative_nasal_energy",
        )
        metrics = {
            name: _readonly_array(getattr(self, name), np.float64)
            for name in metric_names
        }
        singular_values = _readonly_array(self.singular_values, np.float64)

        if coefficient_basis.ndim != 2:
            raise ValueError("coefficient_basis must have shape (S, rank)")
        mode_count, rank = coefficient_basis.shape
        if rank != int(self.retained_rank):
            raise ValueError("coefficient_basis rank does not match retained_rank")
        if vertex_basis.ndim != 3 or vertex_basis.shape[1:] != (3, rank):
            raise ValueError("vertex_basis must have shape (V, 3, rank)")
        if pass_mask.shape != (mode_count,):
            raise ValueError("screening_pass_mask must have shape (S,)")
        if candidate_indices.ndim != 1:
            raise ValueError("candidate_mode_indices must be one-dimensional")
        if len(candidate_indices):
            if (
                candidate_indices[0] < 0
                or candidate_indices[-1] >= mode_count
                or np.any(np.diff(candidate_indices) <= 0)
            ):
                raise ValueError(
                    "candidate_mode_indices must be sorted unique valid indices"
                )
        if not np.array_equal(np.flatnonzero(pass_mask), candidate_indices):
            raise ValueError(
                "candidate_mode_indices and screening_pass_mask are inconsistent"
            )
        if rank > len(candidate_indices):
            raise ValueError("retained_rank must not exceed candidate mode count")
        for name, value in metrics.items():
            if value.shape != (mode_count,) or not np.isfinite(value).all():
                raise ValueError(f"{name} must be a finite array with shape (S,)")
            object.__setattr__(self, name, value)
        if (
            singular_values.ndim != 1
            or len(singular_values) > len(candidate_indices)
            or rank > len(singular_values)
            or not np.isfinite(singular_values).all()
            or np.any(singular_values < 0.0)
            or np.any(np.diff(singular_values) > 1e-12)
        ):
            raise ValueError(
                "singular_values must be finite, non-negative, and descending"
            )
        if not np.isfinite(coefficient_basis).all() or not np.isfinite(
            vertex_basis
        ).all():
            raise ValueError("subspace basis arrays must be finite")
        if rank:
            gram = coefficient_basis.T @ coefficient_basis
            if not np.allclose(gram, np.eye(rank), atol=1e-10, rtol=0.0):
                raise ValueError("coefficient_basis columns must be orthonormal")
        view_names = tuple(str(name) for name in self.view_names)
        metadata = tuple(self.per_view_metadata)
        if view_names != OBSERVABLE_FLAME_VIEW_NAMES:
            raise ValueError("view_names must use canonical subject semantics")
        if (
            len(metadata) != len(view_names)
            or not all(isinstance(item, ProjectionViewMetadata) for item in metadata)
            or tuple(item.name for item in metadata) != view_names
        ):
            raise ValueError("per_view_metadata must match canonical view order")

        object.__setattr__(self, "coefficient_basis", coefficient_basis)
        object.__setattr__(self, "vertex_basis", vertex_basis)
        object.__setattr__(self, "candidate_mode_indices", candidate_indices)
        object.__setattr__(self, "screening_pass_mask", pass_mask)
        object.__setattr__(self, "singular_values", singular_values)
        object.__setattr__(self, "retained_rank", int(self.retained_rank))
        object.__setattr__(self, "view_names", view_names)
        object.__setattr__(self, "per_view_metadata", metadata)
        object.__setattr__(self, "report_data", _freeze_report(self.report_data))

    def to_report_data(self) -> dict:
        """Return an isolated plain structure accepted by ``json.dumps``."""
        return _thaw_report(self.report_data)


def _validate_numeric_array(
    name: str,
    value: np.ndarray,
    shape_tail: tuple[int, ...],
) -> np.ndarray:
    raw = np.asarray(value)
    if raw.ndim != len(shape_tail) + 1 or raw.shape[1:] != shape_tail:
        expected = ", ".join(("N",) + tuple(str(item) for item in shape_tail))
        raise ValueError(f"{name} must have shape ({expected})")
    if not np.issubdtype(raw.dtype, np.number):
        raise ValueError(f"{name} must be numeric")
    result = np.asarray(raw, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError(f"{name} must contain only finite values")
    return result


def _validate_mask(name: str, value: np.ndarray, vertex_count: int) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != (vertex_count,):
        raise ValueError(f"{name} must have shape ({vertex_count},)")
    if not np.issubdtype(mask.dtype, np.bool_):
        raise ValueError(f"{name} must be a boolean array")
    return np.asarray(mask, dtype=bool)


def _validate_shape_basis(
    shape_basis: np.ndarray,
    vertex_count: int,
) -> np.ndarray:
    raw = np.asarray(shape_basis)
    if raw.ndim == 3 and raw.shape[:2] == (vertex_count, 3):
        volumetric = raw
    elif raw.ndim == 2 and raw.shape[0] == vertex_count * 3:
        volumetric = raw.reshape(vertex_count, 3, raw.shape[1])
    else:
        raise ValueError(
            "shape_basis must have shape (V, 3, S) or flattened (3V, S)"
        )
    if volumetric.shape[2] < 1:
        raise ValueError("shape_basis must contain at least one mode")
    if not np.issubdtype(volumetric.dtype, np.number):
        raise ValueError("shape_basis must be numeric")
    result = np.asarray(volumetric, dtype=np.float64)
    if not np.isfinite(result).all():
        raise ValueError("shape_basis must contain only finite values")
    return result


def _canonical_views(views: Sequence[ProjectionView]) -> tuple[ProjectionView, ...]:
    try:
        supplied = tuple(views)
    except TypeError as exc:
        raise ValueError(
            "views must contain exactly three ProjectionView values"
        ) from exc
    if len(supplied) != 3 or not all(
        isinstance(view, ProjectionView) for view in supplied
    ):
        raise ValueError("views must contain exactly three ProjectionView values")
    by_name = {view.name: view for view in supplied}
    if len(by_name) != 3 or set(by_name) != set(OBSERVABLE_FLAME_VIEW_NAMES):
        raise ValueError(
            "views must contain exactly front, subject-left, and subject-right"
        )
    return tuple(by_name[name] for name in OBSERVABLE_FLAME_VIEW_NAMES)


def perspective_projection_jacobian(
    vertices: np.ndarray,
    vertex_basis: np.ndarray,
    view: ProjectionView,
    *,
    min_depth: float = 1e-6,
) -> np.ndarray:
    """Return analytic pixel derivatives with rows ``(u0,v0,u1,v1,...)``."""
    if not isinstance(view, ProjectionView):
        raise ValueError("view must be a ProjectionView")
    points = _validate_numeric_array("vertices", vertices, (3,))
    basis = np.asarray(vertex_basis)
    if (
        basis.ndim != 3
        or basis.shape[:2] != (len(points), 3)
        or not np.issubdtype(basis.dtype, np.number)
    ):
        raise ValueError("vertex_basis must have shape (V, 3, M)")
    basis = np.asarray(basis, dtype=np.float64)
    if not np.isfinite(basis).all():
        raise ValueError("vertex_basis must contain only finite values")
    depth_limit = _finite_real("min_depth", min_depth, minimum=0.0)
    if depth_limit <= 0.0:
        raise ValueError("min_depth must be greater than zero")

    camera_points = points @ view.R_model_to_camera.T + view.t_model_to_camera
    if np.any(camera_points[:, 2] <= depth_limit):
        raise ValueError(
            f"nasal support must have positive depth above min_depth in {view.name}"
        )
    homogeneous = camera_points @ view.K.T
    denominator = homogeneous[:, 2]
    if np.any(denominator <= depth_limit):
        raise ValueError(
            f"nasal support has invalid perspective denominator in {view.name}"
        )

    denominator_squared = denominator[:, None] ** 2
    projection_derivative = np.empty((len(points), 2, 3), dtype=np.float64)
    projection_derivative[:, 0, :] = (
        view.K[0][None, :] * denominator[:, None]
        - homogeneous[:, 0, None] * view.K[2][None, :]
    ) / denominator_squared
    projection_derivative[:, 1, :] = (
        view.K[1][None, :] * denominator[:, None]
        - homogeneous[:, 1, None] * view.K[2][None, :]
    ) / denominator_squared
    camera_basis = np.einsum(
        "ij,vjs->vis",
        view.R_model_to_camera,
        basis,
        optimize=True,
    )
    jacobian = np.einsum(
        "vpi,vis->vps",
        projection_derivative,
        camera_basis,
        optimize=True,
    ).reshape(len(points) * 2, basis.shape[2])
    if not np.isfinite(jacobian).all():
        raise ValueError(f"projection Jacobian is non-finite in {view.name}")
    return _readonly_array(jacobian, np.float64)


def _mean_vertex_energy(
    squared_energy: np.ndarray,
    mask: np.ndarray,
) -> np.ndarray:
    if not np.any(mask):
        return np.zeros(squared_energy.shape[1], dtype=np.float64)
    return np.mean(squared_energy[mask], axis=0)


def _screen_modes(
    shape_basis: np.ndarray,
    support_mask: np.ndarray,
    protected_mask: np.ndarray,
    config: ObservableFlameSubspaceConfig,
):
    squared_energy = np.sum(shape_basis * shape_basis, axis=1)
    outside_mask = ~(support_mask | protected_mask)
    nasal_energy = _mean_vertex_energy(squared_energy, support_mask)
    outside_energy = _mean_vertex_energy(squared_energy, outside_mask)
    protected_energy = _mean_vertex_energy(squared_energy, protected_mask)
    epsilon = float(config.screening_epsilon)

    response_denominator = nasal_energy + outside_energy
    nasal_response_ratio = np.divide(
        nasal_energy,
        response_denominator,
        out=np.zeros_like(nasal_energy),
        where=response_denominator > epsilon,
    )
    safe_nasal = np.maximum(nasal_energy, epsilon)
    outside_to_nasal = outside_energy / safe_nasal
    protected_to_nasal = protected_energy / safe_nasal
    strongest_nasal = float(np.max(nasal_energy))
    if strongest_nasal <= epsilon:
        relative_nasal = np.zeros_like(nasal_energy)
    else:
        relative_nasal = nasal_energy / strongest_nasal

    pass_mask = (
        (nasal_energy > epsilon)
        & (
            nasal_response_ratio
            >= float(config.min_nasal_response_ratio)
        )
        & (
            protected_to_nasal
            <= float(config.max_protected_to_nasal_energy_ratio)
        )
        & (
            relative_nasal
            >= float(config.relative_nasal_energy_floor)
        )
    )
    return {
        "nasal_mean_squared_energy": nasal_energy,
        "outside_mean_squared_energy": outside_energy,
        "protected_mean_squared_energy": protected_energy,
        "nasal_response_ratio": nasal_response_ratio,
        "outside_to_nasal_energy_ratio": outside_to_nasal,
        "protected_to_nasal_energy_ratio": protected_to_nasal,
        "relative_nasal_energy": relative_nasal,
        "pass_mask": pass_mask,
    }


def _balance_view_jacobian(
    pixel_jacobian: np.ndarray,
    intrinsic: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    row_count = int(pixel_jacobian.shape[0])
    normalization = np.linalg.inv(intrinsic[:2, :2])
    normalized = np.einsum(
        "ij,vjs->vis",
        normalization,
        pixel_jacobian.reshape(row_count // 2, 2, pixel_jacobian.shape[1]),
        optimize=True,
    ).reshape(pixel_jacobian.shape)
    row_scale = 1.0 / np.sqrt(float(row_count))
    return normalized * row_scale, normalization, row_scale


def _numerical_rank(
    values: np.ndarray,
    config: ObservableFlameSubspaceConfig,
) -> int:
    if values.size == 0 or values.shape[1] == 0:
        return 0
    singular_values = np.linalg.svd(values, compute_uv=False)
    if not len(singular_values):
        return 0
    threshold = max(
        float(singular_values[0])
        * float(config.relative_singular_value_threshold),
        float(config.singular_value_epsilon),
    )
    return int(np.count_nonzero(singular_values > threshold))


def _canonicalize_columns(values: np.ndarray) -> np.ndarray:
    canonical = np.array(values, dtype=np.float64, copy=True)
    for column in range(canonical.shape[1]):
        pivot = int(np.argmax(np.abs(canonical[:, column])))
        if canonical[pivot, column] < 0.0:
            canonical[:, column] *= -1.0
    return canonical


def build_observable_flame_subspace(
    vertices: np.ndarray,
    shape_basis: np.ndarray,
    support_mask: np.ndarray,
    protected_mask: np.ndarray,
    views: Sequence[ProjectionView],
    config: Optional[ObservableFlameSubspaceConfig] = None,
) -> ObservableFlameSubspaceResult:
    """Screen FLAME modes and retain the fixed three-view observable subspace.

    Each per-vertex two-row pixel block is premultiplied by the inverse 2x2
    intrinsic linear block, including focal anisotropy and skew, then divided
    by ``sqrt(2 * support_vertex_count)``. This gives a resolution-neutral
    averaged normalized-camera-coordinate response without amplifying a
    geometrically weak view to unit norm.
    """
    cfg = ObservableFlameSubspaceConfig() if config is None else config
    if not isinstance(cfg, ObservableFlameSubspaceConfig):
        raise ValueError("config must be an ObservableFlameSubspaceConfig")
    points = _validate_numeric_array("vertices", vertices, (3,))
    if len(points) < 1:
        raise ValueError("vertices must contain at least one vertex")
    basis = _validate_shape_basis(shape_basis, len(points))
    support = _validate_mask("support_mask", support_mask, len(points))
    protected = _validate_mask("protected_mask", protected_mask, len(points))
    if not np.any(support):
        raise ValueError("support_mask must contain at least one nasal vertex")
    if np.any(support & protected):
        raise ValueError("support_mask and protected_mask must be disjoint")
    canonical_views = _canonical_views(views)

    metrics = _screen_modes(basis, support, protected, cfg)
    candidate_indices = np.flatnonzero(metrics["pass_mask"])
    support_points = points[support]
    candidate_vertex_basis = basis[support][:, :, candidate_indices]
    view_blocks = []
    metadata = []
    for view in canonical_views:
        raw_jacobian = perspective_projection_jacobian(
            support_points,
            candidate_vertex_basis,
            view,
            min_depth=float(cfg.min_depth),
        )
        row_count = int(raw_jacobian.shape[0])
        balanced, normalization, row_scale = _balance_view_jacobian(
            raw_jacobian,
            view.K,
        )
        view_blocks.append(balanced)
        metadata.append(
            ProjectionViewMetadata(
                name=view.name,
                row_count=row_count,
                support_vertex_count=int(np.count_nonzero(support)),
                intrinsic_normalization_matrix=normalization,
                row_scale=row_scale,
                pixel_frobenius_norm=float(np.linalg.norm(raw_jacobian)),
                balanced_frobenius_norm=float(np.linalg.norm(balanced)),
                rank=_numerical_rank(balanced, cfg),
            )
        )

    combined_jacobian = np.vstack(view_blocks)
    mode_count = basis.shape[2]
    if not len(candidate_indices):
        singular_values = np.zeros(0, dtype=np.float64)
        coefficient_basis = np.zeros((mode_count, 0), dtype=np.float64)
        status = "no_candidate_modes"
    else:
        _left, singular_values, right_t = np.linalg.svd(
            combined_jacobian,
            full_matrices=False,
        )
        largest = float(singular_values[0]) if len(singular_values) else 0.0
        threshold = max(
            largest * float(cfg.relative_singular_value_threshold),
            float(cfg.singular_value_epsilon),
        )
        retained_rank = min(
            int(np.count_nonzero(singular_values > threshold)),
            int(cfg.max_rank),
        )
        candidate_coefficients = _canonicalize_columns(
            right_t[:retained_rank].T
        )
        coefficient_basis = np.zeros(
            (mode_count, retained_rank),
            dtype=np.float64,
        )
        coefficient_basis[candidate_indices] = candidate_coefficients
        status = "ok" if retained_rank else "no_observable_rank"

    retained_rank = int(coefficient_basis.shape[1])
    vertex_basis = np.einsum(
        "vcs,sr->vcr",
        basis,
        coefficient_basis,
        optimize=True,
    )
    report = {
        "schema": "observable-flame-subspace-v2",
        "status": status,
        "config": asdict(cfg),
        "view_names": list(OBSERVABLE_FLAME_VIEW_NAMES),
        "vertex_count": int(len(points)),
        "support_vertex_count": int(np.count_nonzero(support)),
        "protected_vertex_count": int(np.count_nonzero(protected)),
        "original_mode_count": int(mode_count),
        "candidate_mode_count": int(len(candidate_indices)),
        "candidate_mode_indices": [int(item) for item in candidate_indices],
        "retained_rank": retained_rank,
        "view_validation": {
            "rotation_orthogonality_tolerance": float(_ROTATION_TOLERANCE),
            "rotation_determinant_tolerance": float(_ROTATION_TOLERANCE),
            "accepted_rotation_handling": "nearest proper SO(3) via SVD",
            "intrinsic_final_row_tolerance": float(_INTRINSIC_TOLERANCE),
        },
        "screening": {
            "energy_normalization": "mean per-vertex squared displacement",
            "outside_excludes_protected": True,
            "nasal_mean_squared_energy": [
                float(item)
                for item in metrics["nasal_mean_squared_energy"]
            ],
            "outside_mean_squared_energy": [
                float(item)
                for item in metrics["outside_mean_squared_energy"]
            ],
            "protected_mean_squared_energy": [
                float(item)
                for item in metrics["protected_mean_squared_energy"]
            ],
            "nasal_response_ratio": [
                float(item) for item in metrics["nasal_response_ratio"]
            ],
            "outside_to_nasal_energy_ratio": [
                float(item)
                for item in metrics["outside_to_nasal_energy_ratio"]
            ],
            "protected_to_nasal_energy_ratio": [
                float(item)
                for item in metrics["protected_to_nasal_energy_ratio"]
            ],
            "relative_nasal_energy": [
                float(item) for item in metrics["relative_nasal_energy"]
            ],
            "pass_mask": [bool(item) for item in metrics["pass_mask"]],
        },
        "row_balancing": (
            "each 2-row vertex block premultiplied by the inverse 2x2 "
            "intrinsic linear block, then divided by "
            "sqrt(2*support_vertex_count)"
        ),
        "singular_values": [float(item) for item in singular_values],
        "views": [asdict(item) for item in metadata],
    }
    return ObservableFlameSubspaceResult(
        coefficient_basis=coefficient_basis,
        vertex_basis=vertex_basis,
        candidate_mode_indices=candidate_indices,
        screening_pass_mask=metrics["pass_mask"],
        nasal_mean_squared_energy=metrics["nasal_mean_squared_energy"],
        outside_mean_squared_energy=metrics["outside_mean_squared_energy"],
        protected_mean_squared_energy=metrics[
            "protected_mean_squared_energy"
        ],
        nasal_response_ratio=metrics["nasal_response_ratio"],
        outside_to_nasal_energy_ratio=metrics[
            "outside_to_nasal_energy_ratio"
        ],
        protected_to_nasal_energy_ratio=metrics[
            "protected_to_nasal_energy_ratio"
        ],
        relative_nasal_energy=metrics["relative_nasal_energy"],
        singular_values=singular_values,
        retained_rank=retained_rank,
        view_names=OBSERVABLE_FLAME_VIEW_NAMES,
        per_view_metadata=tuple(metadata),
        config=cfg,
        report_data=report,
    )

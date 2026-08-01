"""Robust 3D fitting of the semantic alar basis to RoMa observations."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping

import numpy as np
from scipy.optimize import least_squares


def _readonly(value: Any, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


@dataclass(frozen=True)
class RoMaNasalFitConfig:
    coefficient_bound: float = 2.5
    coefficient_prior_weight: float = 0.12
    nuisance_translation_bound_mm: float = 30.0
    orientation_weight: float = 25.0
    minimum_orientation_ratio: float = 0.55
    robust_f_scale: float = 1.0
    max_nfev: int = 800

    def __post_init__(self) -> None:
        for name in (
            "coefficient_bound",
            "coefficient_prior_weight",
            "nuisance_translation_bound_mm",
            "orientation_weight",
            "minimum_orientation_ratio",
            "robust_f_scale",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if float(self.minimum_orientation_ratio) >= 1.0:
            raise ValueError("minimum_orientation_ratio must be below one")
        if isinstance(self.max_nfev, bool) or int(self.max_nfev) < 1:
            raise ValueError("max_nfev must be positive")


@dataclass(frozen=True)
class RoMaNasalFitResult:
    success: bool
    coefficients: np.ndarray
    candidate_vertices: np.ndarray
    initial_rmse_mm: float
    final_rmse_mm: float
    observation_count: int
    optimizer_message: str
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        vertices = np.asarray(self.candidate_vertices, dtype=np.float64)
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError("coefficients must be a finite vector")
        if vertices.ndim != 2 or vertices.shape[1] != 3 or not np.isfinite(vertices).all():
            raise ValueError("candidate_vertices must have finite shape (V, 3)")
        object.__setattr__(self, "coefficients", _readonly(coefficients, np.float64))
        object.__setattr__(self, "candidate_vertices", _readonly(vertices, np.float64))
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "success": bool(self.success),
            "coefficients": self.coefficients.tolist(),
            "initial_rmse_mm": float(self.initial_rmse_mm),
            "final_rmse_mm": float(self.final_rmse_mm),
            "observation_count": int(self.observation_count),
            "optimizer_message": self.optimizer_message,
            "metadata": dict(self.metadata),
        }


def fit_roma_nasal_surface(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    basis_vectors: np.ndarray,
    attachment_face_indices: np.ndarray,
    attachment_barycentric: np.ndarray,
    target_points_model: np.ndarray,
    observation_weights: np.ndarray,
    *,
    config: RoMaNasalFitConfig | None = None,
) -> RoMaNasalFitResult:
    """Fit semantic coefficients to exact baseline surface attachments."""
    limits = config or RoMaNasalFitConfig()
    baseline = np.asarray(baseline_vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    vectors = np.asarray(basis_vectors, dtype=np.float64)
    face_indices = np.asarray(attachment_face_indices, dtype=np.int64).reshape(-1)
    barycentric = np.asarray(attachment_barycentric, dtype=np.float64)
    targets = np.asarray(target_points_model, dtype=np.float64)
    weights = np.asarray(observation_weights, dtype=np.float64).reshape(-1)
    if baseline.ndim != 2 or baseline.shape[1] != 3 or not np.isfinite(baseline).all():
        raise ValueError("baseline_vertices must have finite shape (V, 3)")
    if topology.ndim != 2 or topology.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if vectors.ndim != 3 or vectors.shape[1:] != baseline.shape:
        raise ValueError("basis_vectors must have shape (M, V, 3)")
    count = len(face_indices)
    if count < 6:
        raise ValueError("at least six surface observations are required")
    if barycentric.shape != (count, 3) or targets.shape != (count, 3):
        raise ValueError("attachment and target arrays disagree")
    if weights.shape != (count,):
        raise ValueError("observation_weights must have shape (N,)")
    if (
        np.any(face_indices < 0)
        or np.any(face_indices >= len(topology))
        or not np.isfinite(barycentric).all()
        or np.any(barycentric < -1e-8)
        or not np.allclose(np.sum(barycentric, axis=1), 1.0, atol=1e-6)
        or not np.isfinite(targets).all()
        or not np.isfinite(weights).all()
        or np.any(weights <= 0.0)
    ):
        raise ValueError("surface observations contain invalid values")

    triangles = topology[face_indices]
    baseline_points = np.sum(
        baseline[triangles] * barycentric[:, :, None],
        axis=1,
    )
    sampled_modes = np.einsum(
        "mnvc,nv->mnc",
        vectors[:, triangles, :],
        barycentric,
        optimize=True,
    )
    scale = float(np.max(np.linalg.norm(vectors, axis=2)))
    if not np.isfinite(scale) or scale <= 1e-8:
        raise ValueError("basis_vectors have no usable displacement scale")
    normalized_weights = np.sqrt(weights / max(float(np.median(weights)), 1e-8))
    active_vertices = np.any(np.abs(vectors) > 0.0, axis=(0, 2))
    orientation_faces = topology[np.any(active_vertices[topology], axis=1)]
    reference_first = (
        baseline[orientation_faces[:, 1]] - baseline[orientation_faces[:, 0]]
    )
    reference_second = (
        baseline[orientation_faces[:, 2]] - baseline[orientation_faces[:, 0]]
    )
    reference_cross = np.cross(reference_first, reference_second)
    reference_norm2 = np.einsum(
        "ij,ij->i",
        reference_cross,
        reference_cross,
    )
    valid_orientation = reference_norm2 > 1e-20
    orientation_faces = orientation_faces[valid_orientation]
    reference_cross = reference_cross[valid_orientation]
    reference_inverse_norm2 = 1.0 / reference_norm2[valid_orientation]

    mode_count = vectors.shape[0]

    def unpack(parameters: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        coefficients = parameters[:mode_count]
        nuisance_translation = parameters[mode_count:] * scale
        return coefficients, nuisance_translation

    def observation_residual(parameters: np.ndarray) -> np.ndarray:
        coefficients, nuisance_translation = unpack(parameters)
        predicted = baseline_points + np.einsum(
            "m,mnc->nc",
            coefficients,
            sampled_modes,
            optimize=True,
        )
        predicted += nuisance_translation
        return ((predicted - targets) / scale * normalized_weights[:, None]).reshape(-1)

    def residual(parameters: np.ndarray) -> np.ndarray:
        coefficients, _nuisance_translation = unpack(parameters)
        prior = np.sqrt(float(limits.coefficient_prior_weight)) * coefficients
        candidate = baseline + np.einsum(
            "m,mvc->vc",
            coefficients,
            vectors,
            optimize=True,
        )
        first = (
            candidate[orientation_faces[:, 1]]
            - candidate[orientation_faces[:, 0]]
        )
        second = (
            candidate[orientation_faces[:, 2]]
            - candidate[orientation_faces[:, 0]]
        )
        cross = np.cross(first, second)
        orientation_ratios = (
            np.einsum("ij,ij->i", cross, reference_cross)
            * reference_inverse_norm2
        )
        orientation = float(limits.orientation_weight) * np.maximum(
            0.0,
            float(limits.minimum_orientation_ratio) - orientation_ratios,
        )
        return np.r_[observation_residual(parameters), prior, orientation]

    initial_translation = np.median(targets - baseline_points, axis=0)
    translation_bound = float(limits.nuisance_translation_bound_mm) / 1000.0
    initial_translation = np.clip(
        initial_translation,
        -translation_bound,
        translation_bound,
    )
    initial = np.r_[
        np.zeros(mode_count, dtype=np.float64),
        initial_translation / scale,
    ]
    lower = np.r_[
        np.full(mode_count, -float(limits.coefficient_bound)),
        np.full(3, -translation_bound / scale),
    ]
    upper = np.r_[
        np.full(mode_count, float(limits.coefficient_bound)),
        np.full(3, translation_bound / scale),
    ]
    solved = least_squares(
        residual,
        initial,
        bounds=(lower, upper),
        method="trf",
        loss="soft_l1",
        f_scale=float(limits.robust_f_scale),
        max_nfev=int(limits.max_nfev),
        x_scale="jac",
        ftol=1e-10,
        xtol=1e-10,
        gtol=1e-10,
    )
    coefficients, nuisance_translation = unpack(
        np.asarray(solved.x, dtype=np.float64)
    )
    displacement = np.einsum(
        "m,mvc->vc",
        coefficients,
        vectors,
        optimize=True,
    )
    candidate = baseline + displacement
    candidate_first = (
        candidate[orientation_faces[:, 1]] - candidate[orientation_faces[:, 0]]
    )
    candidate_second = (
        candidate[orientation_faces[:, 2]] - candidate[orientation_faces[:, 0]]
    )
    candidate_cross = np.cross(candidate_first, candidate_second)
    final_orientation_ratios = (
        np.einsum("ij,ij->i", candidate_cross, reference_cross)
        * reference_inverse_norm2
    )
    initial_error = observation_residual(initial).reshape(-1, 3) * scale
    final_error = observation_residual(solved.x).reshape(-1, 3) * scale
    raw_initial_error = baseline_points - targets
    initial_rmse = float(np.sqrt(np.mean(np.sum(initial_error**2, axis=1))) * 1000.0)
    final_rmse = float(np.sqrt(np.mean(np.sum(final_error**2, axis=1))) * 1000.0)
    success = bool(
        solved.success
        and np.isfinite(candidate).all()
        and final_rmse <= initial_rmse
    )
    return RoMaNasalFitResult(
        success=success,
        coefficients=coefficients,
        candidate_vertices=candidate,
        initial_rmse_mm=initial_rmse,
        final_rmse_mm=final_rmse,
        observation_count=count,
        optimizer_message=str(solved.message),
        metadata={
            "basis_scale_mm": scale * 1000.0,
            "coefficient_bound": float(limits.coefficient_bound),
            "nuisance_translation_mm": (nuisance_translation * 1000.0).tolist(),
            "nuisance_translation_bound_mm": float(
                limits.nuisance_translation_bound_mm
            ),
            "minimum_orientation_ratio": float(np.min(final_orientation_ratios)),
            "orientation_face_count": int(len(orientation_faces)),
            "raw_initial_rmse_mm": float(
                np.sqrt(np.mean(np.sum(raw_initial_error**2, axis=1))) * 1000.0
            ),
            "nfev": int(solved.nfev),
            "cost": float(solved.cost),
        },
    )


__all__ = [
    "RoMaNasalFitConfig",
    "RoMaNasalFitResult",
    "fit_roma_nasal_surface",
]

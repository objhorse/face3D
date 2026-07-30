"""Compact semantic deformation basis for the nasal base.

The basis deliberately excludes the bridge, broad nasal tip, outer alar
width, and the rest of the face. It is a fixed-topology local model intended
to refine nostril apertures and the columella after the broader nose fit.
"""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from src.geometry.nasal_semantic_basis import (
    NasalSemanticFrame,
    _balance_paired_weights,
    _compact_falloff,
    _distances_from_seeds,
    _landmark_points,
    _readonly_array,
    _semantic_frame,
    _surface_graph,
    _validate_build_inputs,
    _validate_rotation,
)


NASAL_BASE_MODE_NAMES = (
    "columella_vertical",
    "columella_depth",
    "nostril_width_shared",
    "nostril_width_asymmetry",
    "nostril_height_shared",
    "nostril_height_asymmetry",
    "alar_rim_curvature_shared",
    "alar_rim_curvature_asymmetry",
)

__all__ = [
    "NASAL_BASE_MODE_NAMES",
    "NasalBaseSemanticBasis",
    "NasalBaseSemanticBasisConfig",
    "apply_nasal_base_semantic_basis",
    "build_nasal_base_semantic_basis",
]

# Only the two inner nostril rims and the columella are editable landmarks.
# All other landmark triangles remain bit-exactly fixed.
_EDITABLE_LANDMARK_INDICES = np.asarray((32, 33, 34), dtype=np.int64)
_PROTECTED_LANDMARK_INDICES = np.asarray(
    tuple(index for index in range(68) if index not in set(_EDITABLE_LANDMARK_INDICES)),
    dtype=np.int64,
)


@dataclass(frozen=True)
class NasalBaseSemanticBasisConfig:
    """Subject-relative dimensions of the compact nasal-base support."""

    support_radius_ratio: float = 0.075
    inner_rim_radius_ratio: float = 0.052
    columella_radius_ratio: float = 0.046
    unit_displacement_ratio: float = 0.006
    support_epsilon: float = 1e-8

    def __post_init__(self) -> None:
        for name in (
            "support_radius_ratio",
            "inner_rim_radius_ratio",
            "columella_radius_ratio",
            "unit_displacement_ratio",
            "support_epsilon",
        ):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if self.inner_rim_radius_ratio > self.support_radius_ratio:
            raise ValueError("inner_rim_radius_ratio must not exceed support radius")
        if self.columella_radius_ratio > self.support_radius_ratio:
            raise ValueError("columella_radius_ratio must not exceed support radius")
        if self.unit_displacement_ratio >= self.support_radius_ratio:
            raise ValueError("unit displacement must be smaller than support radius")


@dataclass(frozen=True)
class NasalBaseSemanticBasis:
    """Eight immutable displacement fields restricted to the nasal base."""

    names: tuple[str, ...]
    vectors: np.ndarray
    weights: np.ndarray
    mode_support_masks: np.ndarray
    protected_mask: np.ndarray
    support_mask: np.ndarray
    region_masks: Mapping[str, np.ndarray]
    semantic_frame: NasalSemanticFrame
    face_width: float
    unit_scale: float
    config: NasalBaseSemanticBasisConfig
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(self, "vectors", _readonly_array(self.vectors, np.float64))
        object.__setattr__(self, "weights", _readonly_array(self.weights, np.float64))
        object.__setattr__(
            self,
            "mode_support_masks",
            _readonly_array(self.mode_support_masks, bool),
        )
        object.__setattr__(
            self,
            "protected_mask",
            _readonly_array(self.protected_mask, bool),
        )
        object.__setattr__(
            self,
            "support_mask",
            _readonly_array(self.support_mask, bool),
        )
        object.__setattr__(
            self,
            "region_masks",
            MappingProxyType(
                {
                    str(name): _readonly_array(mask, bool)
                    for name, mask in dict(self.region_masks).items()
                }
            ),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        self.validate()

    def validate(self, vertex_count: Optional[int] = None) -> None:
        if self.names != NASAL_BASE_MODE_NAMES:
            raise ValueError("nasal-base semantic mode names or order are invalid")
        if self.vectors.ndim != 3 or self.vectors.shape[0] != 8:
            raise ValueError("vectors must have shape (8, V, 3)")
        mode_count, inferred_vertex_count, coordinates = self.vectors.shape
        if coordinates != 3:
            raise ValueError("vectors must have shape (8, V, 3)")
        if vertex_count is not None and inferred_vertex_count != int(vertex_count):
            raise ValueError("basis vertex count does not match vertices")
        if self.weights.shape != (mode_count, inferred_vertex_count):
            raise ValueError("weights must have shape (8, V)")
        if self.mode_support_masks.shape != (mode_count, inferred_vertex_count):
            raise ValueError("mode support masks must have shape (8, V)")
        if self.protected_mask.shape != (inferred_vertex_count,):
            raise ValueError("protected mask does not match vertex count")
        if self.support_mask.shape != (inferred_vertex_count,):
            raise ValueError("support mask does not match vertex count")
        if not np.isfinite(self.vectors).all() or not np.isfinite(self.weights).all():
            raise ValueError("basis arrays must be finite")
        if np.any(self.weights < 0.0) or np.any(self.weights > 1.0 + 1e-12):
            raise ValueError("basis weights must lie in [0, 1]")
        if np.any(self.support_mask & self.protected_mask):
            raise ValueError("editable support must exclude protected vertices")
        if np.any(self.vectors[:, self.protected_mask, :] != 0.0):
            raise ValueError("protected vertices must have exactly zero displacement")
        if np.any(self.vectors[:, ~self.support_mask, :] != 0.0):
            raise ValueError("vectors must be zero outside nasal-base support")
        active = np.linalg.norm(self.vectors, axis=2) > 0.0
        if not np.array_equal(active, self.mode_support_masks):
            raise ValueError("mode support masks must match nonzero vectors")
        if not np.array_equal(active, self.weights > 0.0):
            raise ValueError("weights must match nonzero vectors")
        if not np.isfinite(self.face_width) or self.face_width <= 0.0:
            raise ValueError("face_width must be finite and positive")
        if not np.isfinite(self.unit_scale) or self.unit_scale <= 0.0:
            raise ValueError("unit_scale must be finite and positive")


def _normalized_field(
    field: np.ndarray,
    unit_scale: float,
    epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    magnitudes = np.linalg.norm(field, axis=1)
    peak = float(np.max(magnitudes))
    if not np.isfinite(peak) or peak <= epsilon:
        raise ValueError("nasal-base semantic mode has empty support")
    normalized = field / peak
    weights = np.linalg.norm(normalized, axis=1)
    active = weights > epsilon
    vectors = normalized * float(unit_scale)
    vectors[~active] = 0.0
    weights[~active] = 0.0
    return vectors, weights, active


def build_nasal_base_semantic_basis(
    vertices: np.ndarray,
    faces: np.ndarray,
    lmk_tri_vidx: np.ndarray,
    lmk_bary_coords: np.ndarray,
    model_to_front_camera: np.ndarray,
    config: Optional[NasalBaseSemanticBasisConfig] = None,
) -> NasalBaseSemanticBasis:
    """Build a deterministic eight-mode basis around the nostril apertures."""
    cfg = NasalBaseSemanticBasisConfig() if config is None else config
    if not isinstance(cfg, NasalBaseSemanticBasisConfig):
        raise ValueError("config must be a NasalBaseSemanticBasisConfig")
    verts, faces_np, landmark_triangles, barycentric = _validate_build_inputs(
        vertices,
        faces,
        lmk_tri_vidx,
        lmk_bary_coords,
    )
    rotation = _validate_rotation(model_to_front_camera)
    landmarks = _landmark_points(verts, landmark_triangles, barycentric)
    frame, face_width = _semantic_frame(landmarks, rotation)

    protected_vertices = np.unique(
        landmark_triangles[_PROTECTED_LANDMARK_INDICES].reshape(-1)
    )
    protected_mask = np.zeros(len(verts), dtype=bool)
    protected_mask[protected_vertices] = True

    right_seeds = np.asarray(landmark_triangles[32], dtype=np.int64)
    center_seeds = np.asarray(landmark_triangles[33], dtype=np.int64)
    left_seeds = np.asarray(landmark_triangles[34], dtype=np.int64)
    graph = _surface_graph(verts, faces_np)
    right_distance = _distances_from_seeds(graph, right_seeds)
    center_distance = _distances_from_seeds(graph, center_seeds)
    left_distance = _distances_from_seeds(graph, left_seeds)
    global_distance = np.minimum(
        np.minimum(right_distance, center_distance),
        left_distance,
    )
    support_mask = (
        global_distance <= face_width * float(cfg.support_radius_ratio)
    ) & ~protected_mask

    right_weight = _compact_falloff(
        right_distance,
        face_width * float(cfg.inner_rim_radius_ratio),
    )
    center_weight = _compact_falloff(
        center_distance,
        face_width * float(cfg.columella_radius_ratio),
    )
    left_weight = _compact_falloff(
        left_distance,
        face_width * float(cfg.inner_rim_radius_ratio),
    )
    for weight in (right_weight, center_weight, left_weight):
        weight[~support_mask] = 0.0
    left_weight, right_weight = _balance_paired_weights(
        left_weight,
        right_weight,
    )

    left_axis = frame.subject_left
    up_axis = frame.up
    depth_axis = frame.depth
    paired_weight = np.clip(left_weight + right_weight, 0.0, 1.0)
    rim_direction = depth_axis - 0.35 * up_axis
    rim_direction = rim_direction / np.linalg.norm(rim_direction)

    raw_modes = np.stack(
        (
            center_weight[:, None] * up_axis,
            center_weight[:, None] * depth_axis,
            (
                left_weight[:, None] * left_axis
                - right_weight[:, None] * left_axis
            ),
            (
                left_weight[:, None] * left_axis
                + right_weight[:, None] * left_axis
            ),
            paired_weight[:, None] * up_axis,
            (
                left_weight[:, None] * up_axis
                - right_weight[:, None] * up_axis
            ),
            paired_weight[:, None] * rim_direction,
            (
                left_weight[:, None] * rim_direction
                - right_weight[:, None] * rim_direction
            ),
        ),
        axis=0,
    )
    raw_modes[:, protected_mask | ~support_mask, :] = 0.0

    unit_scale = face_width * float(cfg.unit_displacement_ratio)
    vectors = []
    weights = []
    masks = []
    for field in raw_modes:
        vector, weight, active = _normalized_field(
            field,
            unit_scale,
            float(cfg.support_epsilon),
        )
        vectors.append(vector)
        weights.append(weight)
        masks.append(active)

    return NasalBaseSemanticBasis(
        names=NASAL_BASE_MODE_NAMES,
        vectors=np.stack(vectors, axis=0),
        weights=np.stack(weights, axis=0),
        mode_support_masks=np.stack(masks, axis=0),
        protected_mask=protected_mask,
        support_mask=support_mask,
        region_masks={
            "subject_right_inner_rim": right_weight > cfg.support_epsilon,
            "columella": center_weight > cfg.support_epsilon,
            "subject_left_inner_rim": left_weight > cfg.support_epsilon,
        },
        semantic_frame=frame,
        face_width=face_width,
        unit_scale=unit_scale,
        config=cfg,
        metadata={
            "editable_landmarks": tuple(
                int(index) for index in _EDITABLE_LANDMARK_INDICES
            ),
            "protected_landmarks": tuple(
                int(index) for index in _PROTECTED_LANDMARK_INDICES
            ),
            "distance_metric": "mesh-edge physical geodesic",
            "outer_alar_policy": "bit-exactly fixed",
        },
    )


def apply_nasal_base_semantic_basis(
    baseline_vertices: np.ndarray,
    basis: NasalBaseSemanticBasis,
    coefficients: np.ndarray,
) -> np.ndarray:
    """Apply dimensionless coefficients without changing mesh topology."""
    baseline_value = np.asarray(baseline_vertices)
    if (
        baseline_value.ndim != 2
        or baseline_value.shape[1] != 3
        or not np.issubdtype(baseline_value.dtype, np.floating)
    ):
        raise ValueError("vertices must be a floating array with shape (V, 3)")
    if not np.isfinite(baseline_value).all():
        raise ValueError("vertices must contain only finite values")
    basis.validate(len(baseline_value))
    coefficient_value = np.asarray(coefficients, dtype=np.float64)
    if coefficient_value.shape != (len(NASAL_BASE_MODE_NAMES),):
        raise ValueError("coefficients must have shape (8,)")
    if not np.isfinite(coefficient_value).all():
        raise ValueError("coefficients must contain only finite values")

    baseline = np.array(baseline_value, copy=True)
    if not np.any(coefficient_value):
        return baseline
    displacement = np.einsum(
        "m,mvc->vc",
        coefficient_value,
        basis.vectors,
        optimize=True,
    )
    displacement[basis.protected_mask] = 0.0
    return baseline + displacement.astype(baseline.dtype, copy=False)

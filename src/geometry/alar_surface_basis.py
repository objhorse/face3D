"""Compact subject-relative deformation basis for the outer nasal wings."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from src.geometry.nasal_semantic_basis import (
    NasalSemanticBasis,
    NasalSemanticBasisConfig,
    NasalSemanticFrame,
    build_nasal_semantic_basis,
)


ALAR_SURFACE_MODE_NAMES = (
    "alar_width_shared",
    "alar_width_asymmetry",
    "alar_flare_shared",
    "alar_flare_asymmetry",
    "alar_vertical_shared",
    "alar_rim_curvature_shared",
)

_PROTECTED_LANDMARK_INDICES = np.r_[0:31, 33, 36:68]


def _readonly(value: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _landmark_points(
    vertices: np.ndarray,
    triangles: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    return np.sum(
        vertices[triangles] * barycentric[:, :, None],
        axis=1,
    )


def _normalized_mode(
    field: np.ndarray,
    support: np.ndarray,
    unit_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(field, dtype=np.float64).copy()
    values[~support] = 0.0
    magnitude = np.linalg.norm(values, axis=1)
    peak = float(np.max(magnitude))
    if not np.isfinite(peak) or peak <= 1e-12:
        raise ValueError("outer-alar semantic mode has empty support")
    values *= float(unit_scale) / peak
    active = np.linalg.norm(values, axis=1) > 1e-14
    values[~active] = 0.0
    return values, active


@dataclass(frozen=True)
class AlarSurfaceBasis:
    names: tuple[str, ...]
    vectors: np.ndarray
    mode_support_masks: np.ndarray
    support_mask: np.ndarray
    protected_mask: np.ndarray
    region_masks: Mapping[str, np.ndarray]
    semantic_frame: NasalSemanticFrame
    face_width: float
    unit_scale: float
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(str(value) for value in self.names))
        object.__setattr__(self, "vectors", _readonly(self.vectors, np.float64))
        object.__setattr__(
            self,
            "mode_support_masks",
            _readonly(self.mode_support_masks, bool),
        )
        object.__setattr__(self, "support_mask", _readonly(self.support_mask, bool))
        object.__setattr__(
            self,
            "protected_mask",
            _readonly(self.protected_mask, bool),
        )
        object.__setattr__(
            self,
            "region_masks",
            MappingProxyType(
                {
                    str(name): _readonly(mask, bool)
                    for name, mask in dict(self.region_masks).items()
                }
            ),
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        self.validate()

    def validate(self, vertex_count: Optional[int] = None) -> None:
        if self.names != ALAR_SURFACE_MODE_NAMES:
            raise ValueError("outer-alar basis mode ordering is invalid")
        if self.vectors.ndim != 3 or self.vectors.shape[0] != len(self.names):
            raise ValueError("outer-alar vectors must have shape (6, V, 3)")
        if self.vectors.shape[2] != 3:
            raise ValueError("outer-alar vectors must have shape (6, V, 3)")
        count = self.vectors.shape[1]
        if vertex_count is not None and count != int(vertex_count):
            raise ValueError("outer-alar basis vertex count mismatch")
        if self.mode_support_masks.shape != (len(self.names), count):
            raise ValueError("outer-alar mode masks must have shape (6, V)")
        for name, mask in (
            ("support", self.support_mask),
            ("protected", self.protected_mask),
        ):
            if mask.shape != (count,):
                raise ValueError(f"outer-alar {name} mask has invalid shape")
        if not np.isfinite(self.vectors).all():
            raise ValueError("outer-alar basis contains non-finite vectors")
        if np.any(self.support_mask & self.protected_mask):
            raise ValueError("outer-alar support overlaps protected vertices")
        if np.any(self.mode_support_masks & ~self.support_mask[None, :]):
            raise ValueError("outer-alar mode extends outside support")
        if np.any(self.vectors[:, ~self.support_mask, :] != 0.0):
            raise ValueError("outer-alar vectors must be exact zero outside support")
        if np.any(self.vectors[:, self.protected_mask, :] != 0.0):
            raise ValueError("protected vertices must remain bit-exact")
        if set(self.region_masks) != {
            "subject_left_alar",
            "subject_right_alar",
            "transition",
        }:
            raise ValueError("outer-alar region masks are incomplete")
        if not np.isfinite(self.face_width) or float(self.face_width) <= 0.0:
            raise ValueError("face_width must be finite and positive")
        if not np.isfinite(self.unit_scale) or float(self.unit_scale) <= 0.0:
            raise ValueError("unit_scale must be finite and positive")


def build_alar_surface_basis(
    vertices: np.ndarray,
    faces: np.ndarray,
    lmk_tri_vidx: np.ndarray,
    lmk_bary_coords: np.ndarray,
    model_to_front_camera: np.ndarray,
    *,
    unit_displacement_ratio: float = 0.009,
    projection_basis: NasalSemanticBasis | None = None,
    projection_config: NasalSemanticBasisConfig | None = None,
) -> AlarSurfaceBasis:
    """Build six outer-wing modes while preserving the accepted A2 center."""
    verts = np.asarray(vertices, dtype=np.float64)
    topology = np.asarray(faces, dtype=np.int64)
    triangles = np.asarray(lmk_tri_vidx, dtype=np.int64)
    barycentric = np.asarray(lmk_bary_coords, dtype=np.float64)
    if verts.ndim != 2 or verts.shape[1] != 3 or not np.isfinite(verts).all():
        raise ValueError("vertices must have finite shape (V, 3)")
    if topology.ndim != 2 or topology.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if triangles.shape != (68, 3) or barycentric.shape != (68, 3):
        raise ValueError("FLAME landmark embedding must have shape (68, 3)")
    ratio = float(unit_displacement_ratio)
    if not np.isfinite(ratio) or not 0.002 <= ratio <= 0.02:
        raise ValueError("unit_displacement_ratio must lie in [0.002, 0.02]")

    broad = (
        build_nasal_semantic_basis(
            verts,
            topology,
            triangles,
            barycentric,
            model_to_front_camera,
            config=projection_config,
        )
        if projection_basis is None
        else projection_basis
    )
    if not isinstance(broad, NasalSemanticBasis):
        raise ValueError("projection_basis must be a NasalSemanticBasis")
    broad.validate(len(verts))

    landmarks = _landmark_points(verts, triangles, barycentric)
    local = (verts - broad.semantic_frame.origin) @ broad.semantic_frame.matrix.T
    left_seed = landmarks[35]
    right_seed = landmarks[31]
    left_seed_local = (
        left_seed - broad.semantic_frame.origin
    ) @ broad.semantic_frame.matrix.T
    right_seed_local = (
        right_seed - broad.semantic_frame.origin
    ) @ broad.semantic_frame.matrix.T

    protected_vertices = np.unique(
        triangles[_PROTECTED_LANDMARK_INDICES].reshape(-1)
    )
    protected = np.array(broad.protected_mask, dtype=bool, copy=True)
    protected[protected_vertices] = True

    left_core = np.array(
        broad.region_masks["subject_left_nose_wing"],
        dtype=bool,
        copy=True,
    )
    right_core = np.array(
        broad.region_masks["subject_right_nose_wing"],
        dtype=bool,
        copy=True,
    )
    transition = np.array(
        broad.region_masks["tip_alar_transition"],
        dtype=bool,
        copy=True,
    )
    lateral_limit = 0.24 * max(
        abs(float(left_seed_local[0])),
        abs(float(right_seed_local[0])),
    )
    transition &= np.abs(local[:, 0]) >= lateral_limit
    left_region = (left_core | (transition & (local[:, 0] >= 0.0))) & ~protected
    right_region = (
        right_core | (transition & (local[:, 0] < 0.0))
    ) & ~protected
    support = (left_region | right_region) & ~protected

    broad_weight = np.max(
        np.asarray(broad.weights, dtype=np.float64)[0:4],
        axis=0,
    )
    transition_weight = np.asarray(broad.weights, dtype=np.float64)[7]
    envelope = np.maximum(broad_weight, 0.55 * transition_weight)
    envelope[~support] = 0.0
    if float(np.max(envelope)) <= 1e-12:
        raise ValueError("projection basis provides no editable outer-alar support")
    envelope /= float(np.max(envelope))
    left_weight = envelope * left_region
    right_weight = envelope * right_region

    left_axis = broad.semantic_frame.subject_left
    up_axis = broad.semantic_frame.up
    depth_axis = broad.semantic_frame.depth
    shared = left_weight + right_weight
    seed_height = np.where(
        left_region,
        float(left_seed_local[1]),
        float(right_seed_local[1]),
    )
    lower = np.clip(
        (
            seed_height
            + 0.018 * float(broad.face_width)
            - local[:, 1]
        )
        / (0.075 * float(broad.face_width)),
        0.0,
        1.0,
    )
    lower *= support
    rim_direction = depth_axis - 0.42 * up_axis
    rim_direction /= np.linalg.norm(rim_direction)

    raw = (
        (
            left_weight[:, None] * left_axis
            - right_weight[:, None] * left_axis
        ),
        (
            left_weight[:, None] * left_axis
            + right_weight[:, None] * left_axis
        ),
        shared[:, None] * depth_axis,
        (
            left_weight[:, None] * depth_axis
            - right_weight[:, None] * depth_axis
        ),
        shared[:, None] * up_axis,
        (shared * lower)[:, None] * rim_direction,
    )
    unit_scale = float(broad.face_width) * ratio
    vectors = []
    masks = []
    for field in raw:
        vector, active = _normalized_mode(field, support, unit_scale)
        vectors.append(vector)
        masks.append(active)
    stacked = np.stack(vectors, axis=0)
    stacked[:, protected, :] = 0.0
    stacked[:, ~support, :] = 0.0

    return AlarSurfaceBasis(
        names=ALAR_SURFACE_MODE_NAMES,
        vectors=stacked,
        mode_support_masks=np.stack(masks, axis=0),
        support_mask=support,
        protected_mask=protected,
        region_masks={
            "subject_left_alar": left_region,
            "subject_right_alar": right_region,
            "transition": transition & support,
        },
        semantic_frame=broad.semantic_frame,
        face_width=float(broad.face_width),
        unit_scale=unit_scale,
        metadata={
            "parameterization": "six-dimensional outer-alar semantic basis",
            "projection_region_source": "canonical broad nasal region masks",
            "protected_landmarks": tuple(
                int(value) for value in _PROTECTED_LANDMARK_INDICES
            ),
            "outer_alar_policy": "editable from multiview surface evidence",
            "outside_support_policy": "bit-exactly fixed",
        },
    )


def apply_alar_surface_basis(
    baseline_vertices: np.ndarray,
    basis: AlarSurfaceBasis,
    coefficients: np.ndarray,
) -> np.ndarray:
    baseline = np.asarray(baseline_vertices)
    basis.validate(len(baseline))
    values = np.asarray(coefficients, dtype=np.float64)
    if values.shape != (len(ALAR_SURFACE_MODE_NAMES),):
        raise ValueError("outer-alar coefficients must have shape (6,)")
    if not np.isfinite(values).all():
        raise ValueError("outer-alar coefficients must be finite")
    candidate = np.array(baseline, copy=True)
    if np.any(values):
        displacement = np.einsum(
            "m,mvc->vc",
            values,
            basis.vectors,
            optimize=True,
        )
        displacement[basis.protected_mask | ~basis.support_mask] = 0.0
        candidate += displacement.astype(candidate.dtype, copy=False)
    return candidate


__all__ = [
    "ALAR_SURFACE_MODE_NAMES",
    "AlarSurfaceBasis",
    "apply_alar_surface_basis",
    "build_alar_surface_basis",
]

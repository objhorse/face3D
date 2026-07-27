"""Deterministic low-dimensional semantic deformations for a fixed-topology nose."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from src.geometry.semantic_regions import build_default_nose_mouth_control_seeds


NASAL_SEMANTIC_MODE_NAMES = (
    "alar_width_shared",
    "alar_width_asymmetry",
    "alar_depth_shared",
    "alar_depth_asymmetry",
    "tip_depth",
    "tip_vertical",
    "tip_roundness",
    "tip_alar_fullness",
)

__all__ = [
    "NASAL_SEMANTIC_MODE_NAMES",
    "NasalSemanticBasis",
    "NasalSemanticBasisConfig",
    "NasalSemanticFrame",
    "apply_nasal_semantic_basis",
    "build_nasal_semantic_basis",
]

_PROTECTED_LANDMARK_INDICES = np.r_[0:17, 33, 36:48, 48:68]
_NASAL_SEED_NAMES = (
    "nose_bridge",
    "nose_tip",
    "subject_left_nose_wing",
    "subject_right_nose_wing",
)
_REGION_MASK_NAMES = _NASAL_SEED_NAMES + ("tip_alar_transition",)
_ROTATION_TOLERANCE = 1e-5


def _readonly_array(value: np.ndarray, dtype=None) -> np.ndarray:
    """Snapshot an array into an immutable bytes-backed NumPy view."""
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    backing = contiguous.tobytes(order="C")
    return np.frombuffer(
        backing,
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _readonly_region_masks(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    try:
        items = dict(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("region_masks must be a mapping") from exc
    return MappingProxyType(
        {str(name): _readonly_array(value, dtype=bool) for name, value in items.items()}
    )


def _readonly_seed_indices(
    values: Mapping[str, np.ndarray],
) -> Mapping[str, np.ndarray]:
    try:
        items = dict(values)
    except (TypeError, ValueError) as exc:
        raise ValueError("seed_indices must be a mapping") from exc
    result = {}
    for name, value in items.items():
        indices = np.asarray(value)
        if (
            indices.ndim != 1
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ValueError(
                f"seed index '{name}' must be a non-empty integer 1-D array"
            )
        result[str(name)] = _readonly_array(indices, dtype=np.int64)
    return MappingProxyType(result)


@dataclass(frozen=True)
class NasalSemanticBasisConfig:
    """Subject-independent ratios controlling compact support and unit amplitude."""

    support_radius_ratio: float = 0.18
    bridge_radius_ratio: float = 0.145
    wing_radius_ratio: float = 0.125
    tip_radius_ratio: float = 0.115
    unit_displacement_ratio: float = 0.012
    support_epsilon: float = 1e-8
    orthogonalization_passes: int = 8
    max_pairwise_weighted_cosine: float = 0.20

    def __post_init__(self) -> None:
        ratios = {
            "support_radius_ratio": self.support_radius_ratio,
            "bridge_radius_ratio": self.bridge_radius_ratio,
            "wing_radius_ratio": self.wing_radius_ratio,
            "tip_radius_ratio": self.tip_radius_ratio,
            "unit_displacement_ratio": self.unit_displacement_ratio,
        }
        for name, value in ratios.items():
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Real)
                or not np.isfinite(value)
                or float(value) <= 0.0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.support_epsilon, (bool, np.bool_))
            or not isinstance(self.support_epsilon, Real)
            or not np.isfinite(self.support_epsilon)
            or not 0.0 < self.support_epsilon < 1e-3
        ):
            raise ValueError("support_epsilon must be finite and in (0, 1e-3)")
        for name in ("bridge_radius_ratio", "wing_radius_ratio", "tip_radius_ratio"):
            if float(getattr(self, name)) > float(self.support_radius_ratio):
                raise ValueError(f"{name} must not exceed support_radius_ratio")
        if float(self.unit_displacement_ratio) >= float(self.support_radius_ratio):
            raise ValueError(
                "unit_displacement_ratio must be smaller than support_radius_ratio"
            )
        if (
            isinstance(self.orthogonalization_passes, (bool, np.bool_))
            or not isinstance(self.orthogonalization_passes, (int, np.integer))
            or not 1 <= int(self.orthogonalization_passes) <= 64
        ):
            raise ValueError("orthogonalization_passes must be an integer in [1, 64]")
        if (
            isinstance(self.max_pairwise_weighted_cosine, (bool, np.bool_))
            or not isinstance(self.max_pairwise_weighted_cosine, Real)
            or not np.isfinite(self.max_pairwise_weighted_cosine)
            or not 0.0 < float(self.max_pairwise_weighted_cosine) < 1.0
        ):
            raise ValueError(
                "max_pairwise_weighted_cosine must be finite and in (0, 1)"
            )


@dataclass(frozen=True)
class NasalSemanticFrame:
    """Landmark-derived model-space axes ordered as subject-left, up, and depth."""

    origin: np.ndarray
    matrix: np.ndarray
    model_to_front_camera: np.ndarray

    def __post_init__(self) -> None:
        origin = _readonly_array(self.origin, dtype=np.float64)
        matrix = _readonly_array(self.matrix, dtype=np.float64)
        rotation = _readonly_array(
            _validate_rotation(self.model_to_front_camera),
            dtype=np.float64,
        )
        if origin.shape != (3,) or not np.isfinite(origin).all():
            raise ValueError("semantic frame origin must be a finite 3-vector")
        if matrix.shape != (3, 3) or not np.isfinite(matrix).all():
            raise ValueError("semantic frame matrix must have shape (3, 3)")
        if not np.allclose(matrix @ matrix.T, np.eye(3), atol=1e-7, rtol=0.0):
            raise ValueError("semantic frame axes must be orthonormal")
        _validate_rotation(rotation)
        object.__setattr__(self, "origin", origin)
        object.__setattr__(self, "matrix", matrix)
        object.__setattr__(self, "model_to_front_camera", rotation)

    @property
    def subject_left(self) -> np.ndarray:
        """Unit model-space direction pointing toward the subject's left."""
        return self.matrix[0]

    @property
    def up(self) -> np.ndarray:
        """Unit model-space direction pointing toward the upper face."""
        return self.matrix[1]

    @property
    def depth(self) -> np.ndarray:
        """Unit model-space direction pointing toward the front camera."""
        return self.matrix[2]


@dataclass(frozen=True)
class NasalSemanticBasis:
    """Eight immutable semantic displacement fields on one mesh topology."""

    names: tuple[str, ...]
    vectors: np.ndarray
    weights: np.ndarray
    directions: np.ndarray
    scales: np.ndarray
    mode_support_masks: np.ndarray
    protected_mask: np.ndarray
    support_mask: np.ndarray
    orthogonalization_weights: np.ndarray
    region_masks: Mapping[str, np.ndarray]
    seed_indices: Mapping[str, np.ndarray]
    semantic_frame: NasalSemanticFrame
    face_width: float
    unit_scale: float
    config: NasalSemanticBasisConfig
    metadata: Mapping[str, object]

    def __post_init__(self) -> None:
        if not isinstance(self.semantic_frame, NasalSemanticFrame):
            raise ValueError("semantic_frame must be a NasalSemanticFrame")
        if not isinstance(self.config, NasalSemanticBasisConfig):
            raise ValueError("config must be a NasalSemanticBasisConfig")
        object.__setattr__(self, "names", tuple(str(name) for name in self.names))
        object.__setattr__(self, "vectors", _readonly_array(self.vectors, np.float64))
        object.__setattr__(self, "weights", _readonly_array(self.weights, np.float64))
        object.__setattr__(
            self, "directions", _readonly_array(self.directions, np.float64)
        )
        object.__setattr__(self, "scales", _readonly_array(self.scales, np.float64))
        object.__setattr__(
            self,
            "mode_support_masks",
            _readonly_array(self.mode_support_masks, bool),
        )
        object.__setattr__(
            self, "protected_mask", _readonly_array(self.protected_mask, bool)
        )
        object.__setattr__(self, "support_mask", _readonly_array(self.support_mask, bool))
        object.__setattr__(
            self,
            "orthogonalization_weights",
            _readonly_array(self.orthogonalization_weights, np.float64),
        )
        object.__setattr__(
            self, "region_masks", _readonly_region_masks(self.region_masks)
        )
        object.__setattr__(
            self, "seed_indices", _readonly_seed_indices(self.seed_indices)
        )
        object.__setattr__(self, "metadata", MappingProxyType(dict(self.metadata)))
        self.validate()

    def validate(self, vertex_count: Optional[int] = None) -> None:
        """Raise ``ValueError`` when basis arrays do not share a valid topology."""
        if self.names != NASAL_SEMANTIC_MODE_NAMES:
            raise ValueError("nasal semantic basis names or order are invalid")
        if self.vectors.ndim != 3 or self.vectors.shape[0] != len(self.names):
            raise ValueError("basis vectors must have shape (8, V, 3)")
        mode_count, inferred_vertex_count, coordinate_count = self.vectors.shape
        if coordinate_count != 3:
            raise ValueError("basis vectors must have shape (8, V, 3)")
        if vertex_count is not None and inferred_vertex_count != int(vertex_count):
            raise ValueError("basis vertex count does not match vertices")
        expected_mode_vertex = (mode_count, inferred_vertex_count)
        if self.weights.shape != expected_mode_vertex:
            raise ValueError("basis weights must have shape (8, V)")
        if self.directions.shape != self.vectors.shape:
            raise ValueError("basis directions must have shape (8, V, 3)")
        if self.mode_support_masks.shape != expected_mode_vertex:
            raise ValueError("mode support masks must have shape (8, V)")
        if self.scales.shape != (mode_count,):
            raise ValueError("basis scales must have shape (8,)")
        if self.protected_mask.shape != (inferred_vertex_count,):
            raise ValueError("protected mask does not match basis vertex count")
        if self.support_mask.shape != (inferred_vertex_count,):
            raise ValueError("support mask does not match basis vertex count")
        if self.orthogonalization_weights.shape != (inferred_vertex_count,):
            raise ValueError(
                "orthogonalization weights do not match basis vertex count"
            )
        for name, value in (
            ("vectors", self.vectors),
            ("weights", self.weights),
            ("directions", self.directions),
            ("scales", self.scales),
            ("orthogonalization weights", self.orthogonalization_weights),
        ):
            if not np.isfinite(value).all():
                raise ValueError(f"basis {name} contain non-finite values")
        if np.any(self.weights < 0.0) or np.any(self.weights > 1.0 + 1e-12):
            raise ValueError("basis weights must lie in [0, 1]")
        if np.any(self.mode_support_masks & ~self.support_mask[None, :]):
            raise ValueError("mode support extends outside nasal support")
        if np.any(self.support_mask & self.protected_mask):
            raise ValueError("nasal support must exclude protected vertices")
        if np.any(
            (self.orthogonalization_weights < 0.0)
            | (self.orthogonalization_weights > 1.0 + 1e-12)
        ):
            raise ValueError("orthogonalization weights must lie in [0, 1]")
        if np.any(
            self.orthogonalization_weights[
                ~self.support_mask | self.protected_mask
            ]
            != 0.0
        ):
            raise ValueError(
                "orthogonalization weights must be zero outside editable support"
            )
        if np.any(self.vectors[:, self.protected_mask, :] != 0.0):
            raise ValueError("protected vertices must have exactly zero displacement")
        if np.any(self.weights[:, self.protected_mask] != 0.0):
            raise ValueError("protected vertices must have exactly zero weight")
        if not np.isfinite(self.face_width) or float(self.face_width) <= 0.0:
            raise ValueError("face_width must be finite and positive")
        if not np.isfinite(self.unit_scale) or float(self.unit_scale) <= 0.0:
            raise ValueError("unit_scale must be finite and positive")
        if not np.allclose(self.scales, self.unit_scale, rtol=0.0, atol=1e-14):
            raise ValueError("all semantic modes must share the configured unit scale")
        active = self.weights > 0.0
        if np.any(active):
            norms = np.linalg.norm(self.directions, axis=2)
            if not np.allclose(norms[active], 1.0, atol=1e-8, rtol=0.0):
                raise ValueError("active basis directions must be normalized")
        vector_active = np.linalg.norm(self.vectors, axis=2) > 0.0
        if not np.array_equal(active, self.mode_support_masks):
            raise ValueError("basis weights and mode support masks are inconsistent")
        if not np.array_equal(vector_active, self.mode_support_masks):
            raise ValueError("basis vectors and mode support masks are inconsistent")
        if np.any(self.directions[~self.mode_support_masks] != 0.0):
            raise ValueError("basis directions must be zero outside mode support")

        if set(self.region_masks) != set(_REGION_MASK_NAMES):
            raise ValueError("region mask keys must match the fixed nasal regions")
        for name in _REGION_MASK_NAMES:
            mask = self.region_masks[name]
            if mask.shape != (inferred_vertex_count,):
                raise ValueError(f"region mask '{name}' has an invalid shape")
            if np.any(mask & self.protected_mask):
                raise ValueError(f"region mask '{name}' includes protected vertices")
            if np.any(mask & ~self.support_mask):
                raise ValueError(f"region mask '{name}' extends outside nasal support")

        if set(self.seed_indices) != set(_NASAL_SEED_NAMES):
            raise ValueError("seed index keys must match the fixed nasal seeds")
        for name in _NASAL_SEED_NAMES:
            indices = self.seed_indices[name]
            if (
                indices.ndim != 1
                or not len(indices)
                or not np.issubdtype(indices.dtype, np.integer)
            ):
                raise ValueError(
                    f"seed index '{name}' must be a non-empty integer 1-D array"
                )
            if indices.min() < 0 or indices.max() >= inferred_vertex_count:
                raise ValueError(f"seed index '{name}' is outside vertex bounds")
            represented = self.region_masks[name] | self.protected_mask
            if not np.all(represented[indices]):
                raise ValueError(
                    f"seed index '{name}' is inconsistent with region support"
                )
        pairwise_cosine = _pairwise_weighted_cosines(
            self.vectors,
            self.orthogonalization_weights,
        )
        off_diagonal = np.abs(pairwise_cosine - np.eye(mode_count))
        if float(np.max(off_diagonal)) > (
            float(self.config.max_pairwise_weighted_cosine) + 1e-10
        ):
            raise ValueError("basis modes exceed the weighted-cosine redundancy limit")


def _validate_rotation(rotation: np.ndarray) -> np.ndarray:
    try:
        value = np.asarray(rotation, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError("model-to-front-camera rotation must be numeric") from exc
    if value.shape != (3, 3):
        raise ValueError("model-to-front-camera rotation must have shape (3, 3)")
    if not np.isfinite(value).all():
        raise ValueError("model-to-front-camera rotation must be finite")
    rigidity_error = float(np.max(np.abs(value @ value.T - np.eye(3))))
    determinant = float(np.linalg.det(value))
    if rigidity_error > _ROTATION_TOLERANCE:
        raise ValueError(
            "model-to-front-camera rotation is materially non-rigid"
        )
    if determinant <= 0.0 or abs(determinant - 1.0) > _ROTATION_TOLERANCE:
        raise ValueError("model-to-front-camera rotation must have determinant +1")
    left, _singular_values, right_t = np.linalg.svd(value)
    projected = left @ right_t
    if float(np.linalg.det(projected)) <= 0.0:
        raise ValueError("model-to-front-camera rotation must not be a reflection")
    return projected


def _validate_build_inputs(
    vertices: np.ndarray,
    faces: np.ndarray,
    lmk_tri_vidx: np.ndarray,
    lmk_bary_coords: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    verts = np.asarray(vertices)
    if verts.ndim != 2 or verts.shape[1] != 3 or len(verts) < 3:
        raise ValueError("vertices must have shape (V, 3) with at least three vertices")
    if not np.issubdtype(verts.dtype, np.number):
        raise ValueError("vertices must be numeric")
    verts = np.asarray(verts, dtype=np.float64)
    if not np.isfinite(verts).all():
        raise ValueError("vertices must contain only finite values")

    face_values = np.asarray(faces)
    if face_values.ndim != 2 or face_values.shape[1] != 3 or not len(face_values):
        raise ValueError("faces must have non-empty shape (F, 3)")
    if not np.issubdtype(face_values.dtype, np.integer):
        if not np.issubdtype(face_values.dtype, np.number) or not np.equal(
            face_values, np.round(face_values)
        ).all():
            raise ValueError("faces must contain integer vertex indices")
    triangles = np.asarray(face_values, dtype=np.int64)
    if triangles.min() < 0 or triangles.max() >= len(verts):
        raise ValueError("faces contain invalid vertex indices")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 2] == triangles[:, 0])
    ):
        raise ValueError("faces must contain three distinct vertex indices")

    landmark_values = np.asarray(lmk_tri_vidx)
    if landmark_values.shape != (68, 3):
        raise ValueError("lmk_tri_vidx must have shape (68, 3)")
    if not np.issubdtype(landmark_values.dtype, np.integer):
        if not np.issubdtype(landmark_values.dtype, np.number) or not np.equal(
            landmark_values, np.round(landmark_values)
        ).all():
            raise ValueError("lmk_tri_vidx must contain integer vertex indices")
    landmark_triangles = np.asarray(landmark_values, dtype=np.int64)
    if landmark_triangles.min() < 0 or landmark_triangles.max() >= len(verts):
        raise ValueError("lmk_tri_vidx contains invalid vertex indices")
    mesh_face_keys = {tuple(row) for row in np.sort(triangles, axis=1)}
    if any(
        tuple(row) not in mesh_face_keys
        for row in np.sort(landmark_triangles, axis=1)
    ):
        raise ValueError("lmk_tri_vidx rows must correspond to mesh faces")

    barycentric = np.asarray(lmk_bary_coords, dtype=np.float64)
    if barycentric.shape != (68, 3):
        raise ValueError("lmk_bary_coords must have shape (68, 3)")
    if not np.isfinite(barycentric).all():
        raise ValueError("lmk_bary_coords must contain only finite values")
    if np.any(barycentric < -1e-8) or np.any(barycentric > 1.0 + 1e-8):
        raise ValueError("lmk_bary_coords must be valid barycentric weights")
    if not np.allclose(barycentric.sum(axis=1), 1.0, atol=1e-6, rtol=0.0):
        raise ValueError("each lmk_bary_coords row must sum to one")
    return verts, triangles, landmark_triangles, barycentric


def _mesh_edges(faces: np.ndarray) -> np.ndarray:
    edges = np.vstack((faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]))
    edges.sort(axis=1)
    return np.unique(edges, axis=0)


def _surface_graph(vertices: np.ndarray, faces: np.ndarray):
    edges = _mesh_edges(faces)
    lengths = np.linalg.norm(vertices[edges[:, 1]] - vertices[edges[:, 0]], axis=1)
    if not np.isfinite(lengths).all() or np.any(lengths <= 1e-12):
        raise ValueError("mesh edges must have finite positive physical lengths")
    rows = np.concatenate((edges[:, 0], edges[:, 1]))
    columns = np.concatenate((edges[:, 1], edges[:, 0]))
    values = np.concatenate((lengths, lengths))
    return coo_matrix(
        (values, (rows, columns)),
        shape=(len(vertices), len(vertices)),
    ).tocsr()


def _landmark_points(
    vertices: np.ndarray,
    triangles: np.ndarray,
    barycentric: np.ndarray,
) -> np.ndarray:
    return np.sum(vertices[triangles] * barycentric[:, :, None], axis=1)


def _normalize(vector: np.ndarray, description: str) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm <= 1e-10:
        raise ValueError(f"landmarks do not define a usable {description} axis")
    return np.asarray(vector, dtype=np.float64) / norm


def _semantic_frame(
    landmarks: np.ndarray,
    rotation: np.ndarray,
) -> tuple[NasalSemanticFrame, float]:
    depth = _normalize(rotation.T @ np.array([0.0, 0.0, -1.0]), "depth")
    upper_center = np.mean(landmarks[17:27], axis=0)
    lower_center = np.mean(landmarks[6:11], axis=0)
    raw_up = upper_center - lower_center
    up = _normalize(raw_up - float(raw_up @ depth) * depth, "up")

    raw_left = np.mean(landmarks[34:36], axis=0) - np.mean(
        landmarks[31:33], axis=0
    )
    raw_left = raw_left - float(raw_left @ depth) * depth
    raw_left = raw_left - float(raw_left @ up) * up
    subject_left = _normalize(raw_left, "subject-left")
    up = _normalize(
        up - float(up @ depth) * depth - float(up @ subject_left) * subject_left,
        "up",
    )
    frame_matrix = np.stack((subject_left, up, depth), axis=0)

    jaw_projection = landmarks[:17] @ subject_left
    low, high = np.percentile(jaw_projection, [5.0, 95.0])
    face_width = float(high - low)
    if not np.isfinite(face_width) or face_width <= 1e-8:
        raise ValueError("landmarks do not span a robust face width")
    origin = np.mean(landmarks[[30, 33]], axis=0)
    return NasalSemanticFrame(origin, frame_matrix, rotation), face_width


def _distances_from_seeds(graph, seeds: np.ndarray) -> np.ndarray:
    indices = np.unique(np.asarray(seeds, dtype=np.int64))
    distances = dijkstra(graph, indices=indices, directed=False)
    if distances.ndim == 1:
        return np.asarray(distances, dtype=np.float64)
    return np.min(np.asarray(distances, dtype=np.float64), axis=0)


def _compact_falloff(distances: np.ndarray, radius: float) -> np.ndarray:
    """Wendland C2 falloff with value and first derivative zero at the boundary."""
    normalized = np.asarray(distances, dtype=np.float64) / float(radius)
    inside = normalized < 1.0
    result = np.zeros_like(normalized)
    remainder = 1.0 - normalized[inside]
    result[inside] = remainder**4 * (4.0 * normalized[inside] + 1.0)
    return result


def _balance_paired_weights(
    left: np.ndarray,
    right: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Equalize paired weighted L2 mass so shared/asymmetric fields are orthogonal."""
    left_norm = float(np.linalg.norm(left))
    right_norm = float(np.linalg.norm(right))
    if left_norm <= 1e-12 or right_norm <= 1e-12:
        raise ValueError("nasal wing supports are empty after feature protection")
    left_balanced = left / left_norm
    right_balanced = right / right_norm
    peak = max(float(np.max(left_balanced)), float(np.max(right_balanced)))
    return left_balanced / peak, right_balanced / peak


def _support_weighted_inner(
    left: np.ndarray,
    right: np.ndarray,
    support_weights: np.ndarray,
) -> float:
    return float(
        np.sum(
            support_weights[:, None]
            * np.asarray(left, dtype=np.float64)
            * np.asarray(right, dtype=np.float64)
        )
    )


def _pairwise_weighted_cosines(
    modes: np.ndarray,
    support_weights: np.ndarray,
) -> np.ndarray:
    values = np.asarray(modes, dtype=np.float64)
    metric = np.asarray(support_weights, dtype=np.float64)
    gram = np.einsum("v,ivc,jvc->ij", metric, values, values, optimize=True)
    norms = np.sqrt(np.maximum(np.diag(gram), 0.0))
    if np.any(norms <= 1e-14):
        raise ValueError("weighted semantic mode norm is zero")
    return gram / np.outer(norms, norms)


def _deflate_semantic_modes(
    raw_modes: np.ndarray,
    support_weights: np.ndarray,
    support_epsilon: float,
    passes: int,
) -> np.ndarray:
    """Support-weighted MGS constrained to original support and semantic signs."""
    raw = np.asarray(raw_modes, dtype=np.float64)
    raw_magnitudes = np.linalg.norm(raw, axis=2)
    raw_support = raw_magnitudes > float(support_epsilon)
    references = np.zeros_like(raw)
    references[raw_support] = raw[raw_support] / raw_magnitudes[
        raw_support, None
    ]
    deflated: list[np.ndarray] = []
    for mode_index in range(len(raw)):
        current = raw[mode_index].copy()
        mode_support = raw_support[mode_index]
        for _ in range(int(passes)):
            for previous in deflated:
                restricted_previous = previous.copy()
                restricted_previous[~mode_support] = 0.0
                denominator = _support_weighted_inner(
                    restricted_previous,
                    restricted_previous,
                    support_weights,
                )
                if denominator <= 1e-18:
                    continue
                coefficient = _support_weighted_inner(
                    current,
                    restricted_previous,
                    support_weights,
                ) / denominator
                current -= coefficient * restricted_previous

                semantic_alignment = np.sum(
                    current * references[mode_index],
                    axis=1,
                )
                reversed_vertices = semantic_alignment < 0.0
                current[reversed_vertices] -= (
                    semantic_alignment[reversed_vertices, None]
                    * references[mode_index, reversed_vertices]
                )
                current[~mode_support] = 0.0
        semantic_score = _support_weighted_inner(
            current,
            references[mode_index],
            support_weights,
        )
        if semantic_score <= float(support_epsilon):
            raise ValueError(
                f"orthogonalization erased semantic mode "
                f"'{NASAL_SEMANTIC_MODE_NAMES[mode_index]}'"
            )
        deflated.append(current)
    return np.stack(deflated, axis=0)


def _normalized_mode(
    field: np.ndarray,
    unit_scale: float,
    support_epsilon: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    magnitudes = np.linalg.norm(field, axis=1)
    maximum = float(np.max(magnitudes))
    if not np.isfinite(maximum) or maximum <= support_epsilon:
        raise ValueError("semantic mode has empty or degenerate support")
    normalized = field / maximum
    weights = np.linalg.norm(normalized, axis=1)
    active = weights > support_epsilon
    directions = np.zeros_like(normalized)
    directions[active] = normalized[active] / weights[active, None]
    vectors = normalized * float(unit_scale)
    vectors[~active] = 0.0
    weights[~active] = 0.0
    return vectors, weights, directions, active


def build_nasal_semantic_basis(
    vertices: np.ndarray,
    faces: np.ndarray,
    lmk_tri_vidx: np.ndarray,
    lmk_bary_coords: np.ndarray,
    model_to_front_camera: np.ndarray,
    config: Optional[NasalSemanticBasisConfig] = None,
) -> NasalSemanticBasis:
    """Build the fixed eight-mode semantic nasal basis for one mesh frame.

    Landmark ordering follows the standard 68-point convention. Positive
    ``alar_width_asymmetry`` widens the subject-left wing and narrows the
    subject-right wing. Camera depth is transformed back to model space, so
    positive depth always points opposite front-camera +Z.
    """
    cfg = NasalSemanticBasisConfig() if config is None else config
    if not isinstance(cfg, NasalSemanticBasisConfig):
        raise ValueError("config must be a NasalSemanticBasisConfig")
    verts, faces_np, landmark_triangles, barycentric = _validate_build_inputs(
        vertices,
        faces,
        lmk_tri_vidx,
        lmk_bary_coords,
    )
    rotation = _validate_rotation(model_to_front_camera)
    landmarks = _landmark_points(verts, landmark_triangles, barycentric)
    frame, face_width = _semantic_frame(landmarks, rotation)

    all_control_seeds = build_default_nose_mouth_control_seeds(landmark_triangles)
    seed_indices = {
        name: np.asarray(all_control_seeds[name], dtype=np.int64)
        for name in _NASAL_SEED_NAMES
    }
    protected_vertices = np.unique(
        landmark_triangles[_PROTECTED_LANDMARK_INDICES].reshape(-1)
    )
    protected_mask = np.zeros(len(verts), dtype=bool)
    protected_mask[protected_vertices] = True

    graph = _surface_graph(verts, faces_np)
    distances = {
        name: _distances_from_seeds(graph, seeds)
        for name, seeds in seed_indices.items()
    }
    global_distance = np.min(
        np.stack([distances[name] for name in _NASAL_SEED_NAMES], axis=0),
        axis=0,
    )
    global_support = (
        global_distance <= face_width * float(cfg.support_radius_ratio)
    ) & ~protected_mask

    bridge_weight = _compact_falloff(
        distances["nose_bridge"], face_width * float(cfg.bridge_radius_ratio)
    )
    tip_weight = _compact_falloff(
        distances["nose_tip"], face_width * float(cfg.tip_radius_ratio)
    )
    left_wing_weight = _compact_falloff(
        distances["subject_left_nose_wing"],
        face_width * float(cfg.wing_radius_ratio),
    )
    right_wing_weight = _compact_falloff(
        distances["subject_right_nose_wing"],
        face_width * float(cfg.wing_radius_ratio),
    )
    for weight in (bridge_weight, tip_weight, left_wing_weight, right_wing_weight):
        weight[~global_support] = 0.0
    left_wing_weight, right_wing_weight = _balance_paired_weights(
        left_wing_weight, right_wing_weight
    )

    wing_weight = np.clip(left_wing_weight + right_wing_weight, 0.0, 1.0)
    central_tip_weight = tip_weight * (1.0 - 0.65 * wing_weight)
    bridge_transition = np.sqrt(np.maximum(tip_weight * bridge_weight, 0.0))
    vertical_tip_weight = np.maximum(central_tip_weight, 0.30 * bridge_transition)

    side_coordinate = (verts - frame.origin) @ frame.subject_left
    central_gate = np.clip(
        1.0 - np.abs(side_coordinate) / (face_width * 0.25),
        0.0,
        1.0,
    )
    transition_weight = (
        np.sqrt(np.maximum(tip_weight * wing_weight, 0.0))
        * central_gate
        * (1.0 - 0.35 * tip_weight)
        * (1.0 - 0.20 * wing_weight)
    )
    transition_weight[~global_support] = 0.0

    left = frame.subject_left
    up = frame.up
    depth = frame.depth
    alar_width_shared = (
        left_wing_weight[:, None] * left
        - right_wing_weight[:, None] * left
    )
    alar_width_asymmetry = (
        left_wing_weight[:, None] * left
        + right_wing_weight[:, None] * left
    )
    alar_depth_shared = wing_weight[:, None] * depth
    alar_depth_asymmetry = (
        left_wing_weight[:, None] * depth
        - right_wing_weight[:, None] * depth
    )
    tip_depth = central_tip_weight[:, None] * depth
    tip_vertical = vertical_tip_weight[:, None] * up

    radial = verts - frame.origin
    radial = radial - (radial @ depth)[:, None] * depth
    radial_norm = np.linalg.norm(radial, axis=1)
    radial_direction = np.zeros_like(radial)
    radial_valid = radial_norm > 1e-12
    radial_direction[radial_valid] = radial[radial_valid] / radial_norm[
        radial_valid, None
    ]
    radial_ramp = np.clip(
        radial_norm / (face_width * float(cfg.tip_radius_ratio) * 0.45),
        0.0,
        1.0,
    )
    tip_roundness = (
        tip_weight * radial_ramp
    )[:, None] * radial_direction

    outward = np.sign(side_coordinate)[:, None] * left
    fullness_direction = depth + 0.28 * outward
    fullness_direction /= np.linalg.norm(fullness_direction, axis=1, keepdims=True)
    tip_alar_fullness = transition_weight[:, None] * fullness_direction

    raw_modes = np.stack(
        (
            alar_width_shared,
            alar_width_asymmetry,
            alar_depth_shared,
            alar_depth_asymmetry,
            tip_depth,
            tip_vertical,
            tip_roundness,
            tip_alar_fullness,
        ),
        axis=0,
    )
    raw_modes[:, protected_mask | ~global_support, :] = 0.0
    raw_magnitudes = np.linalg.norm(raw_modes, axis=2)
    orthogonalization_weights = np.max(raw_magnitudes, axis=0)
    metric_peak = float(np.max(orthogonalization_weights))
    if metric_peak <= float(cfg.support_epsilon):
        raise ValueError("nasal semantic support is empty after feature protection")
    orthogonalization_weights /= metric_peak
    orthogonalization_weights[protected_mask | ~global_support] = 0.0
    deflated_modes = _deflate_semantic_modes(
        raw_modes,
        orthogonalization_weights,
        float(cfg.support_epsilon),
        int(cfg.orthogonalization_passes),
    )

    unit_scale = face_width * float(cfg.unit_displacement_ratio)
    vectors = []
    weights = []
    directions = []
    mode_support_masks = []
    for raw in deflated_modes:
        vector, weight, direction, active = _normalized_mode(
            raw,
            unit_scale,
            float(cfg.support_epsilon),
        )
        vectors.append(vector)
        weights.append(weight)
        directions.append(direction)
        mode_support_masks.append(active)

    vectors_np = np.stack(vectors, axis=0)
    weights_np = np.stack(weights, axis=0)
    directions_np = np.stack(directions, axis=0)
    mode_support_np = np.stack(mode_support_masks, axis=0)
    support_mask = global_support.copy()
    region_masks = {
        "nose_bridge": (bridge_weight > cfg.support_epsilon) & ~protected_mask,
        "nose_tip": (tip_weight > cfg.support_epsilon) & ~protected_mask,
        "subject_left_nose_wing": (
            left_wing_weight > cfg.support_epsilon
        ) & ~protected_mask,
        "subject_right_nose_wing": (
            right_wing_weight > cfg.support_epsilon
        ) & ~protected_mask,
        "tip_alar_transition": (
            transition_weight > cfg.support_epsilon
        ) & ~protected_mask,
    }
    metadata = {
        "axis_order": ("subject_left", "up", "depth"),
        "depth_convention": "opposite front-camera +Z",
        "asymmetry_sign": (
            "positive widens subject-left and narrows subject-right"
        ),
        "distance_metric": "mesh-edge physical geodesic",
        "normalization": (
            "support-weighted constrained MGS deflation and fixed peak amplitude"
        ),
        "orthogonalization_passes": int(cfg.orthogonalization_passes),
        "max_pairwise_weighted_cosine": float(
            np.max(
                np.abs(
                    _pairwise_weighted_cosines(
                        vectors_np,
                        orthogonalization_weights,
                    )
                    - np.eye(len(NASAL_SEMANTIC_MODE_NAMES))
                )
            )
        ),
        "protected_landmarks": tuple(
            int(index) for index in _PROTECTED_LANDMARK_INDICES
        ),
    }
    return NasalSemanticBasis(
        names=NASAL_SEMANTIC_MODE_NAMES,
        vectors=vectors_np,
        weights=weights_np,
        directions=directions_np,
        scales=np.full(len(NASAL_SEMANTIC_MODE_NAMES), unit_scale),
        mode_support_masks=mode_support_np,
        protected_mask=protected_mask,
        support_mask=support_mask,
        orthogonalization_weights=orthogonalization_weights,
        region_masks=region_masks,
        seed_indices=seed_indices,
        semantic_frame=frame,
        face_width=face_width,
        unit_scale=unit_scale,
        config=cfg,
        metadata=metadata,
    )


def apply_nasal_semantic_basis(
    baseline_vertices: np.ndarray,
    basis: NasalSemanticBasis,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Apply dimensionless coefficients and return ``(candidate, displacement)``.

    The operation is linear and never changes topology. An all-zero coefficient
    vector takes an explicit early return so the candidate is an exact baseline
    copy, including dtype and bit pattern.
    """
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
    if coefficient_value.shape != (len(basis.names),):
        raise ValueError(
            f"coefficients must have shape ({len(basis.names)},)"
        )
    if not np.isfinite(coefficient_value).all():
        raise ValueError("coefficients must contain only finite values")

    baseline = np.array(baseline_value, copy=True)
    if not np.any(coefficient_value):
        return baseline, np.zeros_like(baseline)
    displacement64 = np.einsum(
        "m,mvc->vc",
        coefficient_value,
        basis.vectors,
        optimize=True,
    )
    displacement64[basis.protected_mask] = 0.0
    displacement = displacement64.astype(baseline.dtype, copy=False)
    candidate = baseline + displacement
    return candidate, displacement

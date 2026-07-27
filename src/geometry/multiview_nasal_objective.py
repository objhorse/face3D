"""Immutable candidate meshes and semantic multiview nasal boundary samples."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Real
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import numpy as np

from src.cross_view_geometry import scale_intrinsics
from src.geometry.nasal_observations import (
    NASAL_VIEWS,
    NasalObservationBundle,
)
from src.geometry.nasal_semantic_basis import NASAL_SEMANTIC_MODE_NAMES
from src.geometry.observable_flame_subspace import ProjectionView
from src.appearance.projective_sampling import project_points_strict


__all__ = [
    "CandidateNasalMesh",
    "MultiviewNasalProjection",
    "MultiviewNasalSamplingConfig",
    "ProjectedNasalSamples",
    "build_candidate_nasal_mesh",
    "project_multiview_nasal_boundaries",
]

_REGION_ALIASES = {
    "nose_bridge": ("nose_bridge",),
    "nose_tip": ("nose_tip",),
    "nose_wing_subject_left": (
        "subject_left_nose_wing",
        "nose_wing_subject_left",
    ),
    "nose_wing_subject_right": (
        "subject_right_nose_wing",
        "nose_wing_subject_right",
    ),
    "tip_alar_transition": ("tip_alar_transition",),
}
_FRONT_REGIONS = (
    ("nose_wing_subject_left", "subject-left-alar"),
    ("nose_tip", "nose-tip"),
    ("nose_wing_subject_right", "subject-right-alar"),
)
_SOURCE_PRIORITY = (
    "nose_tip",
    "nose_wing_subject_left",
    "nose_wing_subject_right",
    "tip_alar_transition",
    "nose_bridge",
)


def _readonly_array(value: np.ndarray, dtype=None) -> np.ndarray:
    """Return an immutable bytes-backed snapshot, including through aliases."""
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _finite_real(name: str, value, *, strictly_positive: bool = False) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if result < 0.0 or (strictly_positive and result <= 0.0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


@dataclass(frozen=True)
class MultiviewNasalSamplingConfig:
    """One subject-independent sampling and visibility contract."""

    front_samples_per_region: int = 16
    side_samples_per_view: int = 32
    min_depth: float = 1e-4
    visibility_relative_tolerance: float = 1e-5
    visibility_absolute_tolerance: float = 1e-6
    barycentric_tolerance: float = 1e-5

    def __post_init__(self) -> None:
        for name in ("front_samples_per_region", "side_samples_per_view"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, (int, np.integer))
                or int(value) < 1
            ):
                raise ValueError(f"{name} must be a positive integer")
        _finite_real("min_depth", self.min_depth, strictly_positive=True)
        _finite_real(
            "visibility_relative_tolerance",
            self.visibility_relative_tolerance,
        )
        _finite_real(
            "visibility_absolute_tolerance",
            self.visibility_absolute_tolerance,
        )
        _finite_real("barycentric_tolerance", self.barycentric_tolerance)


def _validate_vertices(value: np.ndarray) -> np.ndarray:
    vertices = np.asarray(value)
    if (
        vertices.ndim != 2
        or vertices.shape[1] != 3
        or len(vertices) < 3
        or not np.issubdtype(vertices.dtype, np.floating)
    ):
        raise ValueError(
            "baseline vertices must be a floating array with shape (V, 3)"
        )
    if not np.isfinite(vertices).all():
        raise ValueError("baseline vertices must contain only finite values")
    return vertices


def _validate_faces(value: np.ndarray, vertex_count: int) -> np.ndarray:
    faces = np.asarray(value)
    if (
        faces.ndim != 2
        or faces.shape[1] != 3
        or not len(faces)
        or not np.issubdtype(faces.dtype, np.integer)
    ):
        raise ValueError("faces must be a non-empty integer array with shape (F, 3)")
    if int(faces.min()) < 0 or int(faces.max()) >= vertex_count:
        raise ValueError("faces contain out-of-range topology indices")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        raise ValueError("each face must contain three distinct topology indices")
    canonical_faces = np.sort(faces.astype(np.int64, copy=False), axis=1)
    if len(np.unique(canonical_faces, axis=0)) != len(faces):
        raise ValueError("faces contain duplicate topology triangles")
    edge_counts = {}
    for triangle in faces:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            edge_counts[edge] = edge_counts.get(edge, 0) + 1
    if any(count > 2 for count in edge_counts.values()):
        raise ValueError("faces contain non-manifold topology edges")
    return faces


@dataclass(frozen=True)
class CandidateNasalMesh:
    """A topology-preserving immutable candidate vertex snapshot."""

    vertices: np.ndarray
    faces: np.ndarray

    def __post_init__(self) -> None:
        vertices = _validate_vertices(self.vertices)
        faces = _validate_faces(self.faces, len(vertices))
        object.__setattr__(
            self,
            "vertices",
            _readonly_array(vertices, vertices.dtype),
        )
        object.__setattr__(self, "faces", _readonly_array(faces, faces.dtype))


def _validate_basis_array(
    name: str,
    value: np.ndarray,
    shape: Tuple[Optional[int], ...],
) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be numeric") from exc
    if array.ndim != len(shape) or any(
        expected is not None and actual != expected
        for actual, expected in zip(array.shape, shape)
    ):
        expected_shape = ", ".join(
            "N" if item is None else str(item) for item in shape
        )
        raise ValueError(f"{name} must have shape ({expected_shape})")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def _validate_coefficients(
    name: str,
    value: np.ndarray,
    length: int,
) -> np.ndarray:
    try:
        coefficients = np.asarray(value, dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} coefficients must be numeric") from exc
    if coefficients.shape != (length,):
        raise ValueError(f"{name} coefficients must have shape ({length},)")
    if not np.isfinite(coefficients).all():
        raise ValueError(f"{name} coefficients must contain only finite values")
    return coefficients


def _semantic_vectors(semantic_basis, vertex_count: int) -> np.ndarray:
    validator = getattr(semantic_basis, "validate", None)
    if callable(validator):
        validator(vertex_count)
    vectors = _validate_basis_array(
        "semantic basis vectors",
        getattr(semantic_basis, "vectors", None),
        (len(NASAL_SEMANTIC_MODE_NAMES), vertex_count, 3),
    )
    if hasattr(semantic_basis, "names") and tuple(semantic_basis.names) != tuple(
        NASAL_SEMANTIC_MODE_NAMES
    ):
        raise ValueError("semantic basis names must use the canonical mode order")
    for name in ("support_mask", "protected_mask"):
        mask = np.asarray(getattr(semantic_basis, name, None))
        if mask.shape != (vertex_count,) or not np.issubdtype(
            mask.dtype, np.bool_
        ):
            raise ValueError(f"semantic basis {name} must be boolean shape (V,)")
    mode_masks = np.asarray(
        getattr(semantic_basis, "mode_support_masks", None)
    )
    if mode_masks.shape != (len(NASAL_SEMANTIC_MODE_NAMES), vertex_count):
        raise ValueError("semantic basis mode_support_masks must have shape (8, V)")
    if not np.issubdtype(mode_masks.dtype, np.bool_):
        raise ValueError("semantic basis mode_support_masks must be boolean")
    return vectors


def build_candidate_nasal_mesh(
    baseline_vertices: np.ndarray,
    faces: np.ndarray,
    observable_flame,
    semantic_basis,
    flame_coefficients: np.ndarray,
    semantic_coefficients: np.ndarray,
) -> CandidateNasalMesh:
    """Combine only fixed-topology FLAME and semantic nasal deformations."""
    baseline = _validate_vertices(baseline_vertices)
    topology = _validate_faces(faces, len(baseline))
    flame_basis = _validate_basis_array(
        "observable FLAME vertex_basis",
        getattr(observable_flame, "vertex_basis", None),
        (len(baseline), 3, None),
    )
    if hasattr(observable_flame, "retained_rank") and int(
        observable_flame.retained_rank
    ) != flame_basis.shape[2]:
        raise ValueError(
            "observable FLAME retained rank does not match vertex_basis"
        )
    semantic_vectors = _semantic_vectors(semantic_basis, len(baseline))
    flame_values = _validate_coefficients(
        "FLAME",
        flame_coefficients,
        flame_basis.shape[2],
    )
    semantic_values = _validate_coefficients(
        "semantic",
        semantic_coefficients,
        semantic_vectors.shape[0],
    )

    candidate = np.array(baseline, copy=True)
    if np.any(flame_values) or np.any(semantic_values):
        flame_displacement = np.einsum(
            "vcr,r->vc",
            flame_basis,
            flame_values,
            optimize=True,
        )
        semantic_displacement = np.einsum(
            "mvc,m->vc",
            semantic_vectors,
            semantic_values,
            optimize=True,
        )
        candidate += flame_displacement.astype(candidate.dtype, copy=False)
        candidate += semantic_displacement.astype(candidate.dtype, copy=False)
        if not np.isfinite(candidate).all():
            raise ValueError("candidate vertices are non-finite")
    return CandidateNasalMesh(candidate, topology)


@dataclass(frozen=True)
class ProjectedNasalSamples:
    """Vectorized residual samples for one canonical semantic view."""

    semantic_view: str
    pixel_xy: np.ndarray
    model_points: np.ndarray
    source_vertex_indices: np.ndarray
    source_weights: np.ndarray
    confidence: np.ndarray
    source_labels: Tuple[str, ...]
    boundary_names: Tuple[str, ...]
    visible: np.ndarray
    depth: np.ndarray

    def __post_init__(self) -> None:
        if self.semantic_view not in NASAL_VIEWS:
            raise ValueError("sample semantic_view is not canonical")
        pixel_xy = np.asarray(self.pixel_xy, dtype=np.float64)
        model_points = np.asarray(self.model_points, dtype=np.float64)
        indices = np.asarray(self.source_vertex_indices)
        weights = np.asarray(self.source_weights, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        visible = np.asarray(self.visible)
        depth = np.asarray(self.depth, dtype=np.float64)
        count = len(pixel_xy)
        expected = {
            "pixel_xy": (count, 2),
            "model_points": (count, 3),
            "source_vertex_indices": (count, 2),
            "source_weights": (count, 2),
            "confidence": (count,),
            "visible": (count,),
            "depth": (count,),
        }
        values = {
            "pixel_xy": pixel_xy,
            "model_points": model_points,
            "source_vertex_indices": indices,
            "source_weights": weights,
            "confidence": confidence,
            "visible": visible,
            "depth": depth,
        }
        for name, shape in expected.items():
            if values[name].shape != shape:
                raise ValueError(f"{name} must have shape {shape}")
        if not np.issubdtype(indices.dtype, np.integer):
            raise ValueError("source_vertex_indices must contain integers")
        if not np.issubdtype(visible.dtype, np.bool_):
            raise ValueError("visible must be boolean")
        numeric = (pixel_xy, model_points, weights, confidence, depth)
        if not all(np.isfinite(value).all() for value in numeric):
            raise ValueError("projected sample arrays must be finite")
        if (
            np.any(indices < 0)
            or np.any(weights < 0.0)
            or not np.allclose(weights.sum(axis=1), 1.0, atol=1e-12, rtol=0.0)
        ):
            raise ValueError("sample edge provenance is invalid")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("sample confidence must lie in [0, 1]")
        if not np.all(visible) or np.any(depth <= 0.0):
            raise ValueError("returned samples must be visible at positive depth")
        labels = tuple(str(value) for value in self.source_labels)
        boundaries = tuple(str(value) for value in self.boundary_names)
        if len(labels) != count or len(boundaries) != count:
            raise ValueError("sample labels must match the sample count")
        object.__setattr__(self, "semantic_view", str(self.semantic_view))
        object.__setattr__(self, "pixel_xy", _readonly_array(pixel_xy, np.float64))
        object.__setattr__(
            self,
            "model_points",
            _readonly_array(model_points, np.float64),
        )
        object.__setattr__(
            self,
            "source_vertex_indices",
            _readonly_array(indices, np.int64),
        )
        object.__setattr__(
            self,
            "source_weights",
            _readonly_array(weights, np.float64),
        )
        object.__setattr__(
            self,
            "confidence",
            _readonly_array(confidence, np.float64),
        )
        object.__setattr__(self, "source_labels", labels)
        object.__setattr__(self, "boundary_names", boundaries)
        object.__setattr__(
            self,
            "visible",
            _readonly_array(visible, bool),
        )
        object.__setattr__(self, "depth", _readonly_array(depth, np.float64))


@dataclass(frozen=True)
class MultiviewNasalProjection:
    """Candidate geometry and ordered immutable samples for all fixed views."""

    candidate_vertices: np.ndarray
    faces: np.ndarray
    per_view: Tuple[ProjectedNasalSamples, ...]

    def __post_init__(self) -> None:
        vertices = _validate_vertices(self.candidate_vertices)
        faces = _validate_faces(self.faces, len(vertices))
        views = tuple(self.per_view)
        if (
            len(views) != len(NASAL_VIEWS)
            or not all(isinstance(value, ProjectedNasalSamples) for value in views)
            or tuple(value.semantic_view for value in views) != tuple(NASAL_VIEWS)
        ):
            raise ValueError("per_view samples must use canonical three-view order")
        object.__setattr__(
            self,
            "candidate_vertices",
            _readonly_array(vertices, vertices.dtype),
        )
        object.__setattr__(self, "faces", _readonly_array(faces, faces.dtype))
        object.__setattr__(self, "per_view", views)

    @property
    def by_view(self) -> Mapping[str, ProjectedNasalSamples]:
        return MappingProxyType(
            {
                samples.semantic_view: samples
                for samples in self.per_view
            }
        )


def _canonical_views(views: Sequence[ProjectionView]) -> Tuple[ProjectionView, ...]:
    try:
        values = tuple(views)
    except TypeError as exc:
        raise ValueError("views must contain three ProjectionView values") from exc
    if (
        len(values) != len(NASAL_VIEWS)
        or not all(isinstance(view, ProjectionView) for view in values)
    ):
        raise ValueError("views must contain three ProjectionView values")
    names = tuple(view.name for view in values)
    if names != tuple(NASAL_VIEWS):
        raise ValueError(
            "ProjectionView names/order must be front, subject-left, subject-right"
        )
    return values


def _region_masks(semantic_basis, vertex_count: int) -> Mapping[str, np.ndarray]:
    _semantic_vectors(semantic_basis, vertex_count)
    raw = getattr(semantic_basis, "region_masks", None)
    if not isinstance(raw, Mapping):
        raise ValueError("semantic basis region_masks must be a mapping")
    result = {}
    support = np.asarray(semantic_basis.support_mask, dtype=bool)
    protected = np.asarray(semantic_basis.protected_mask, dtype=bool)
    if np.any(support & protected):
        raise ValueError("semantic support and protected masks must be disjoint")
    for public_name, aliases in _REGION_ALIASES.items():
        present = [name for name in aliases if name in raw]
        if not present:
            raise ValueError(f"semantic region mask is missing {public_name}")
        mask = np.asarray(raw[present[0]])
        if mask.shape != (vertex_count,) or not np.issubdtype(
            mask.dtype, np.bool_
        ):
            raise ValueError(
                f"semantic region {public_name} must be boolean shape (V,)"
            )
        if any(
            not np.array_equal(mask, np.asarray(raw[name]))
            for name in present[1:]
        ):
            raise ValueError(f"semantic region aliases disagree for {public_name}")
        if np.any(mask & ~support) or np.any(mask & protected):
            raise ValueError(
                f"semantic region {public_name} extends outside editable support"
            )
        result[public_name] = np.asarray(mask, dtype=bool)
    return result


def _edge_adjacency(faces: np.ndarray):
    adjacency = {}
    for face_index, triangle in enumerate(faces):
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            adjacency.setdefault(edge, []).append(face_index)
    return tuple(
        (edge[0], edge[1], tuple(adjacency[edge]))
        for edge in sorted(adjacency)
    )


def _front_region_edges(
    adjacency,
    faces: np.ndarray,
    region_mask: np.ndarray,
) -> Tuple[Tuple[int, int], ...]:
    edges = []
    for first, second, incident in adjacency:
        if not (region_mask[first] and region_mask[second]):
            continue
        if len(incident) == 1 or any(
            not np.all(region_mask[faces[face_index]])
            for face_index in incident
        ):
            edges.append((first, second))
    return tuple(edges)


def _silhouette_edges(
    adjacency,
    camera_points: np.ndarray,
    faces: np.ndarray,
) -> Tuple[Tuple[int, int], ...]:
    triangles = camera_points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centroids = np.mean(triangles, axis=1)
    facing_measure = np.einsum("fi,fi->f", normals, -centroids)
    nondegenerate = np.linalg.norm(normals, axis=1) > 1e-12
    front_facing = facing_measure > 0.0
    result = []
    for first, second, incident in adjacency:
        if len(incident) == 1:
            if nondegenerate[incident[0]]:
                result.append((first, second))
        elif (
            nondegenerate[incident[0]]
            and nondegenerate[incident[1]]
            and front_facing[incident[0]] != front_facing[incident[1]]
        ):
            result.append((first, second))
    return tuple(result)


def _sample_edges(
    edges: Sequence[Tuple[int, int]],
    count: int,
    vertices: np.ndarray,
    projected_vertices: np.ndarray,
):
    usable = []
    for first, second in sorted(edges):
        length = float(
            np.linalg.norm(projected_vertices[second] - projected_vertices[first])
        )
        if np.isfinite(length) and length > 1e-10:
            usable.append((first, second, length))
    if not usable:
        return (
            np.empty((0, 3), dtype=np.float64),
            np.empty((0, 2), dtype=np.int64),
            np.empty((0, 2), dtype=np.float64),
        )
    lengths = np.asarray([item[2] for item in usable], dtype=np.float64)
    cumulative = np.cumsum(lengths)
    targets = (np.arange(count, dtype=np.float64) + 0.5) * (
        cumulative[-1] / float(count)
    )
    segment_indices = np.searchsorted(cumulative, targets, side="right")
    starts = np.concatenate(([0.0], cumulative[:-1]))
    points = []
    indices = []
    weights = []
    for target, segment_index in zip(targets, segment_indices):
        first, second, length = usable[int(segment_index)]
        alpha = float((target - starts[int(segment_index)]) / length)
        points.append(
            (1.0 - alpha) * vertices[first] + alpha * vertices[second]
        )
        indices.append((first, second))
        weights.append((1.0 - alpha, alpha))
    return (
        np.asarray(points, dtype=np.float64),
        np.asarray(indices, dtype=np.int64),
        np.asarray(weights, dtype=np.float64),
    )


def _source_label(
    first: int,
    second: int,
    regions: Mapping[str, np.ndarray],
) -> str:
    best_name = _SOURCE_PRIORITY[-1]
    best_score = -1
    for name in _SOURCE_PRIORITY:
        score = int(regions[name][first]) + int(regions[name][second])
        if score > best_score:
            best_name = name
            best_score = score
    return best_name


def _sparse_visibility(
    sample_pixels: np.ndarray,
    sample_depth: np.ndarray,
    projected_vertices: np.ndarray,
    vertex_depth: np.ndarray,
    faces: np.ndarray,
    config: MultiviewNasalSamplingConfig,
) -> np.ndarray:
    triangle_pixels = projected_vertices[faces]
    triangle_depth = vertex_depth[faces]
    positive = np.all(triangle_depth > float(config.min_depth), axis=1)
    minimum_xy = np.min(triangle_pixels, axis=1)
    maximum_xy = np.max(triangle_pixels, axis=1)
    visible = np.zeros(len(sample_pixels), dtype=bool)
    barycentric_tolerance = float(config.barycentric_tolerance)
    for sample_index, (pixel, depth) in enumerate(
        zip(sample_pixels, sample_depth)
    ):
        candidates = np.flatnonzero(
            positive
            & np.all(pixel >= minimum_xy - barycentric_tolerance, axis=1)
            & np.all(pixel <= maximum_xy + barycentric_tolerance, axis=1)
        )
        nearest = np.inf
        if len(candidates):
            triangles = triangle_pixels[candidates]
            first = triangles[:, 1] - triangles[:, 0]
            second = triangles[:, 2] - triangles[:, 0]
            relative = pixel[None, :] - triangles[:, 0]
            denominator = (
                first[:, 0] * second[:, 1]
                - first[:, 1] * second[:, 0]
            )
            nondegenerate = np.abs(denominator) > 1e-12
            first_weight = np.divide(
                relative[:, 0] * second[:, 1]
                - relative[:, 1] * second[:, 0],
                denominator,
                out=np.full(len(candidates), -np.inf, dtype=np.float64),
                where=nondegenerate,
            )
            second_weight = np.divide(
                first[:, 0] * relative[:, 1]
                - first[:, 1] * relative[:, 0],
                denominator,
                out=np.full(len(candidates), -np.inf, dtype=np.float64),
                where=nondegenerate,
            )
            barycentric = np.column_stack(
                (
                    1.0 - first_weight - second_weight,
                    first_weight,
                    second_weight,
                )
            )
            inside = np.all(
                barycentric >= -barycentric_tolerance,
                axis=1,
            )
            inverse_depth = np.sum(
                barycentric[inside] / triangle_depth[candidates[inside]],
                axis=1,
            )
            valid_depth = inverse_depth[
                (inverse_depth > 0.0) & np.isfinite(inverse_depth)
            ]
            if len(valid_depth):
                nearest = float(np.min(1.0 / valid_depth))
        tolerance = max(
            float(config.visibility_absolute_tolerance),
            abs(float(nearest)) * float(config.visibility_relative_tolerance),
        )
        visible[sample_index] = np.isfinite(nearest) and depth <= nearest + tolerance
    return visible


def _bilinear_confidence(
    confidence: np.ndarray,
    pixels: np.ndarray,
) -> np.ndarray:
    height, width = confidence.shape
    x = pixels[:, 0]
    y = pixels[:, 1]
    x0 = np.floor(x).astype(np.int64)
    y0 = np.floor(y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = x - x0
    wy = y - y0
    source = np.asarray(confidence, dtype=np.float64)
    return (
        source[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + source[y0, x1] * wx * (1.0 - wy)
        + source[y1, x0] * (1.0 - wx) * wy
        + source[y1, x1] * wx * wy
    )


def _project_edge_groups(
    semantic_view: str,
    edge_groups,
    candidate: CandidateNasalMesh,
    observation,
    view: ProjectionView,
    config: MultiviewNasalSamplingConfig,
) -> ProjectedNasalSamples:
    vertex_projection = project_points_strict(
        candidate.vertices,
        view.K,
        view.R_model_to_camera,
        view.t_model_to_camera,
        epsilon=float(config.min_depth),
    )
    all_points = []
    all_indices = []
    all_weights = []
    all_labels = []
    all_boundaries = []
    for edges, count, source_label, boundary_name in edge_groups:
        points, indices, weights = _sample_edges(
            edges,
            count,
            candidate.vertices,
            vertex_projection.pixel_xy,
        )
        all_points.append(points)
        all_indices.append(indices)
        all_weights.append(weights)
        all_labels.extend([source_label] * len(points))
        all_boundaries.extend([boundary_name] * len(points))
    if not all_points or not sum(len(value) for value in all_points):
        raise ValueError(f"no valid nasal samples exist for {semantic_view}")
    model_points = np.vstack(all_points)
    source_indices = np.vstack(all_indices)
    source_weights = np.vstack(all_weights)
    sample_projection = project_points_strict(
        model_points,
        view.K,
        view.R_model_to_camera,
        view.t_model_to_camera,
        epsilon=float(config.min_depth),
    )
    pixels = np.asarray(sample_projection.pixel_xy, dtype=np.float64)
    depth = np.asarray(sample_projection.depth, dtype=np.float64)
    width, height = observation.work_size
    x0, y0, x1, y1 = observation.roi_work_xyxy
    valid = (
        sample_projection.front_facing
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] <= float(width - 1))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] <= float(height - 1))
        & (pixels[:, 0] >= float(x0))
        & (pixels[:, 0] < float(x1))
        & (pixels[:, 1] >= float(y0))
        & (pixels[:, 1] < float(y1))
    )
    visible = np.zeros(len(pixels), dtype=bool)
    selected = np.flatnonzero(valid)
    if len(selected):
        visible[selected] = _sparse_visibility(
            pixels[selected],
            depth[selected],
            vertex_projection.pixel_xy,
            vertex_projection.depth,
            candidate.faces,
            config,
        )
    keep = valid & visible
    if not np.any(keep):
        raise ValueError(f"no valid visible nasal samples exist for {semantic_view}")
    selected_labels = tuple(
        label for label, selected_value in zip(all_labels, keep) if selected_value
    )
    selected_boundaries = tuple(
        name
        for name, selected_value in zip(all_boundaries, keep)
        if selected_value
    )
    selected_pixels = pixels[keep]
    return ProjectedNasalSamples(
        semantic_view=semantic_view,
        pixel_xy=selected_pixels,
        model_points=model_points[keep],
        source_vertex_indices=source_indices[keep],
        source_weights=source_weights[keep],
        confidence=_bilinear_confidence(
            observation.confidence,
            selected_pixels,
        ),
        source_labels=selected_labels,
        boundary_names=selected_boundaries,
        visible=np.ones(np.count_nonzero(keep), dtype=bool),
        depth=depth[keep],
    )


def _validate_work_intrinsics(
    observations: NasalObservationBundle,
    views: Tuple[ProjectionView, ...],
) -> None:
    for view, semantic_view in zip(views, NASAL_VIEWS):
        observation = observations.by_view[semantic_view]
        expected = scale_intrinsics(
            observation.camera.K,
            observation.camera.image_size,
            observation.work_size,
        )
        if not np.allclose(view.K, expected, atol=1e-9, rtol=1e-9):
            raise ValueError(
                f"{semantic_view} ProjectionView K is incompatible with "
                "the observation work-pixel frame"
            )


def project_multiview_nasal_boundaries(
    candidate: CandidateNasalMesh,
    semantic_basis,
    observations: NasalObservationBundle,
    views: Sequence[ProjectionView],
    *,
    config: Optional[MultiviewNasalSamplingConfig] = None,
) -> MultiviewNasalProjection:
    """Project semantic front boundaries and visible side silhouettes."""
    if not isinstance(candidate, CandidateNasalMesh):
        raise ValueError("candidate must be a CandidateNasalMesh")
    if not isinstance(observations, NasalObservationBundle):
        raise ValueError("observations must be a NasalObservationBundle")
    limits = MultiviewNasalSamplingConfig() if config is None else config
    if not isinstance(limits, MultiviewNasalSamplingConfig):
        raise ValueError("config must be a MultiviewNasalSamplingConfig")
    canonical_views = _canonical_views(views)
    _validate_work_intrinsics(observations, canonical_views)
    regions = _region_masks(semantic_basis, len(candidate.vertices))
    adjacency = _edge_adjacency(candidate.faces)
    per_view = []
    for semantic_view, view in zip(NASAL_VIEWS, canonical_views):
        observation = observations.by_view[semantic_view]
        if semantic_view == "front":
            edge_groups = []
            for source_label, boundary_name in _FRONT_REGIONS:
                edges = _front_region_edges(
                    adjacency,
                    candidate.faces,
                    regions[source_label],
                )
                edge_groups.append(
                    (
                        edges,
                        int(limits.front_samples_per_region),
                        source_label,
                        boundary_name,
                    )
                )
        else:
            vertex_projection = project_points_strict(
                candidate.vertices,
                view.K,
                view.R_model_to_camera,
                view.t_model_to_camera,
                epsilon=float(limits.min_depth),
            )
            silhouettes = _silhouette_edges(
                adjacency,
                vertex_projection.camera_points,
                candidate.faces,
            )
            nasal_profile = (
                np.asarray(semantic_basis.support_mask, dtype=bool)
                & (
                    regions["nose_tip"]
                    | regions["tip_alar_transition"]
                    | regions["nose_wing_subject_left"]
                    | regions["nose_wing_subject_right"]
                )
            )
            restricted = tuple(
                edge
                for edge in silhouettes
                if nasal_profile[edge[0]] and nasal_profile[edge[1]]
            )
            by_label = {}
            for edge in restricted:
                label = _source_label(edge[0], edge[1], regions)
                by_label.setdefault(label, []).append(edge)
            ordered_edges = tuple(
                edge
                for label in _SOURCE_PRIORITY
                for edge in sorted(by_label.get(label, ()))
            )
            edge_groups = [
                (
                    ordered_edges,
                    int(limits.side_samples_per_view),
                    "nasal-profile",
                    "nasal-profile",
                )
            ]
        samples = _project_edge_groups(
            semantic_view,
            edge_groups,
            candidate,
            observation,
            view,
            limits,
        )
        if semantic_view != "front":
            side_labels = tuple(
                _source_label(
                    int(indices[0]),
                    int(indices[1]),
                    regions,
                )
                for indices in samples.source_vertex_indices
            )
            samples = ProjectedNasalSamples(
                semantic_view=samples.semantic_view,
                pixel_xy=samples.pixel_xy,
                model_points=samples.model_points,
                source_vertex_indices=samples.source_vertex_indices,
                source_weights=samples.source_weights,
                confidence=samples.confidence,
                source_labels=side_labels,
                boundary_names=samples.boundary_names,
                visible=samples.visible,
                depth=samples.depth,
            )
        per_view.append(samples)
    return MultiviewNasalProjection(
        candidate_vertices=candidate.vertices,
        faces=candidate.faces,
        per_view=tuple(per_view),
    )

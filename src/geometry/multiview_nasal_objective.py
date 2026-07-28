"""Immutable multiview nasal projection and unified objective evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
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
    "MultiviewNasalObjectiveConfig",
    "MultiviewNasalObjectiveContext",
    "MultiviewNasalObjectiveResult",
    "MultiviewNasalResidualEvaluation",
    "MultiviewNasalSoftProjection",
    "MultiviewNasalProjection",
    "MultiviewNasalSamplingConfig",
    "PreparedNasalProjectionContext",
    "ProjectedNasalSamples",
    "SoftProfileNasalSlots",
    "build_candidate_nasal_mesh",
    "build_candidate_nasal_mesh_prepared",
    "evaluate_multiview_nasal_objective",
    "evaluate_multiview_nasal_objective_residuals",
    "prepare_multiview_nasal_objective_context",
    "prepare_nasal_projection_context",
    "project_multiview_nasal_boundaries",
    "project_multiview_nasal_boundaries_prepared",
]

_REGION_NAMES = (
    "nose_bridge",
    "nose_tip",
    "subject_left_nose_wing",
    "subject_right_nose_wing",
    "tip_alar_transition",
)
_FRONT_REGIONS = (
    ("subject_left_nose_wing", "subject-left-alar"),
    ("subject_right_nose_wing", "subject-right-alar"),
)
_SOURCE_PRIORITY = (
    "nose_tip",
    "subject_left_nose_wing",
    "subject_right_nose_wing",
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


@dataclass(frozen=True)
class _PreparedViewTargetContract:
    semantic_view: str
    work_size: Tuple[int, int]
    roi_work_xyxy: Tuple[float, float, float, float]
    confidence: np.ndarray
    target_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.semantic_view not in NASAL_VIEWS:
            raise ValueError("prepared view semantic name is not canonical")
        work_size = tuple(int(value) for value in self.work_size)
        try:
            roi_array = np.asarray(self.roi_work_xyxy, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError("prepared ROI must contain finite coordinates") from exc
        roi = tuple(float(value) for value in roi_array.reshape(-1))
        if (
            len(work_size) != 2
            or min(work_size) < 1
            or roi_array.shape != (4,)
            or not np.isfinite(roi_array).all()
            or not (
                0.0 <= roi[0] < roi[2] <= float(work_size[0])
                and 0.0 <= roi[1] < roi[3] <= float(work_size[1])
            )
        ):
            raise ValueError("prepared work size or ROI is invalid")
        confidence = np.asarray(self.confidence, dtype=np.float64)
        if confidence.shape != (work_size[1], work_size[0]):
            raise ValueError("prepared confidence does not match work_size")
        if not np.isfinite(confidence).all():
            raise ValueError("prepared confidence must contain only finite values")
        target_names = tuple(str(name) for name in self.target_names)
        if not target_names or len(set(target_names)) != len(target_names):
            raise ValueError("prepared target names must be non-empty and unique")
        object.__setattr__(self, "work_size", work_size)
        object.__setattr__(self, "roi_work_xyxy", roi)
        object.__setattr__(
            self,
            "confidence",
            _readonly_array(confidence, np.float64),
        )
        object.__setattr__(self, "target_names", target_names)


@dataclass(frozen=True)
class PreparedNasalProjectionContext:
    """Deeply immutable static inputs reused by finite-difference probes."""

    vertex_count: int
    faces: np.ndarray
    semantic_vectors: np.ndarray
    support_mask: np.ndarray
    protected_mask: np.ndarray
    region_masks: Mapping[str, np.ndarray]
    views: Tuple[ProjectionView, ...]
    view_contracts: Tuple[_PreparedViewTargetContract, ...]
    config: MultiviewNasalSamplingConfig
    edge_vertices: np.ndarray = field(init=False)
    edge_faces: np.ndarray = field(init=False)
    front_loop_vertices: np.ndarray = field(init=False)
    front_loop_offsets: np.ndarray = field(init=False)
    front_loop_source_codes: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        vertex_count = int(self.vertex_count)
        if vertex_count < 3:
            raise ValueError("prepared vertex_count must be at least three")
        faces = _validate_faces(self.faces, vertex_count)
        vectors = _validate_basis_array(
            "prepared semantic vectors",
            self.semantic_vectors,
            (len(NASAL_SEMANTIC_MODE_NAMES), vertex_count, 3),
        )
        support = np.asarray(self.support_mask)
        protected = np.asarray(self.protected_mask)
        for name, mask in (
            ("support_mask", support),
            ("protected_mask", protected),
        ):
            if mask.shape != (vertex_count,) or not np.issubdtype(
                mask.dtype,
                np.bool_,
            ):
                raise ValueError(f"prepared {name} must be boolean shape (V,)")
        if np.any(support & protected):
            raise ValueError("prepared support and protected masks must be disjoint")
        regions = _snapshot_region_masks(self.region_masks, vertex_count)
        for name, mask in regions.items():
            if np.any(mask & protected):
                raise ValueError(
                    f"prepared semantic region {name} intersects protected_mask"
                )
            if np.any(mask & ~support):
                raise ValueError(
                    f"prepared semantic region {name} extends outside support_mask"
                )
        canonical_views = _canonical_views(self.views)
        contracts = tuple(self.view_contracts)
        if (
            len(contracts) != len(NASAL_VIEWS)
            or not all(
                isinstance(contract, _PreparedViewTargetContract)
                for contract in contracts
            )
            or tuple(contract.semantic_view for contract in contracts)
            != tuple(NASAL_VIEWS)
        ):
            raise ValueError(
                "prepared view contracts must use canonical three-view order"
            )
        if not isinstance(self.config, MultiviewNasalSamplingConfig):
            raise ValueError("prepared config must be MultiviewNasalSamplingConfig")

        face_snapshot = _readonly_array(faces, faces.dtype)
        edge_vertices, edge_faces = _build_edge_adjacency_arrays(face_snapshot)
        loop_vertices, loop_offsets, loop_codes = _prepare_front_composite_loops(
            face_snapshot,
            edge_vertices,
            edge_faces,
            regions,
            protected,
        )
        object.__setattr__(self, "vertex_count", vertex_count)
        object.__setattr__(self, "faces", face_snapshot)
        object.__setattr__(
            self,
            "semantic_vectors",
            _readonly_array(vectors, np.float64),
        )
        object.__setattr__(self, "support_mask", _readonly_array(support, bool))
        object.__setattr__(
            self,
            "protected_mask",
            _readonly_array(protected, bool),
        )
        object.__setattr__(self, "region_masks", regions)
        object.__setattr__(self, "views", canonical_views)
        object.__setattr__(self, "view_contracts", contracts)
        object.__setattr__(self, "edge_vertices", edge_vertices)
        object.__setattr__(self, "edge_faces", edge_faces)
        object.__setattr__(self, "front_loop_vertices", loop_vertices)
        object.__setattr__(self, "front_loop_offsets", loop_offsets)
        object.__setattr__(self, "front_loop_source_codes", loop_codes)

    @property
    def observation_target_names(self) -> Tuple[Tuple[str, ...], ...]:
        return tuple(contract.target_names for contract in self.view_contracts)


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
    edge_directions = {}
    for triangle in faces:
        for first, second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            directed = (int(first), int(second))
            edge = tuple(sorted(directed))
            edge_directions.setdefault(edge, []).append(directed)
    if any(len(directions) > 2 for directions in edge_directions.values()):
        raise ValueError("faces contain non-manifold topology edges")
    for directions in edge_directions.values():
        if len(directions) == 2 and directions[0] != (
            directions[1][1],
            directions[1][0],
        ):
            raise ValueError(
                "adjacent faces must traverse shared edges in opposite "
                "directions with consistent winding"
            )
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

    @classmethod
    def _from_validated(
        cls,
        vertices: np.ndarray,
        faces: np.ndarray,
        *,
        reuse_faces: bool,
        reuse_vertices: bool = False,
    ) -> "CandidateNasalMesh":
        validated_vertices = _validate_vertices(vertices)
        instance = object.__new__(cls)
        object.__setattr__(
            instance,
            "vertices",
            (
                validated_vertices
                if reuse_vertices
                else _readonly_array(
                    validated_vertices,
                    validated_vertices.dtype,
                )
            ),
        )
        object.__setattr__(
            instance,
            "faces",
            faces if reuse_faces else _readonly_array(faces, faces.dtype),
        )
        return instance


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
    return CandidateNasalMesh._from_validated(
        candidate,
        topology,
        reuse_faces=False,
    )


def build_candidate_nasal_mesh_prepared(
    baseline_vertices: np.ndarray,
    observable_flame,
    prepared: PreparedNasalProjectionContext,
    flame_coefficients: np.ndarray,
    semantic_coefficients: np.ndarray,
) -> CandidateNasalMesh:
    """Build a candidate while reusing prepared topology and semantics."""
    if not isinstance(prepared, PreparedNasalProjectionContext):
        raise ValueError("prepared must be a PreparedNasalProjectionContext")
    baseline = _validate_vertices(baseline_vertices)
    if len(baseline) != prepared.vertex_count:
        raise ValueError("baseline vertex count does not match prepared context")
    flame_basis = _validate_basis_array(
        "observable FLAME vertex_basis",
        getattr(observable_flame, "vertex_basis", None),
        (prepared.vertex_count, 3, None),
    )
    if hasattr(observable_flame, "retained_rank") and int(
        observable_flame.retained_rank
    ) != flame_basis.shape[2]:
        raise ValueError(
            "observable FLAME retained rank does not match vertex_basis"
        )
    flame_values = _validate_coefficients(
        "FLAME",
        flame_coefficients,
        flame_basis.shape[2],
    )
    semantic_values = _validate_coefficients(
        "semantic",
        semantic_coefficients,
        prepared.semantic_vectors.shape[0],
    )
    candidate = np.array(baseline, copy=True)
    if np.any(flame_values) or np.any(semantic_values):
        candidate += np.einsum(
            "vcr,r->vc",
            flame_basis,
            flame_values,
            optimize=True,
        ).astype(candidate.dtype, copy=False)
        candidate += np.einsum(
            "mvc,m->vc",
            prepared.semantic_vectors,
            semantic_values,
            optimize=True,
        ).astype(candidate.dtype, copy=False)
        if not np.isfinite(candidate).all():
            raise ValueError("candidate vertices are non-finite")
    return CandidateNasalMesh._from_validated(
        candidate,
        prepared.faces,
        reuse_faces=True,
    )


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
        if count < 1:
            raise ValueError("projected view samples must contain at least one sample")
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
        unknown_labels = set(labels) - set(_REGION_NAMES)
        if unknown_labels:
            raise ValueError(
                "source_labels must use canonical semantic region names: "
                + ", ".join(sorted(unknown_labels))
            )
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

    @classmethod
    def _from_canonical_slots(
        cls,
        *,
        semantic_view: str,
        pixel_xy: np.ndarray,
        model_points: np.ndarray,
        source_vertex_indices: np.ndarray,
        source_weights: np.ndarray,
        confidence: np.ndarray,
        source_labels: Tuple[str, ...],
        boundary_names: Tuple[str, ...],
        visible: np.ndarray,
        depth: np.ndarray,
    ) -> "ProjectedNasalSamples":
        """Build fixed objective slots while retaining truthful visibility."""
        visibility = np.asarray(visible)
        if (
            visibility.shape != (len(np.asarray(pixel_xy)),)
            or not np.issubdtype(visibility.dtype, np.bool_)
        ):
            raise ValueError(
                "canonical slot visibility must be boolean shape (N,)"
            )
        validated = cls(
            semantic_view=semantic_view,
            pixel_xy=pixel_xy,
            model_points=model_points,
            source_vertex_indices=source_vertex_indices,
            source_weights=source_weights,
            confidence=confidence,
            source_labels=source_labels,
            boundary_names=boundary_names,
            visible=np.ones(len(visibility), dtype=bool),
            depth=depth,
        )
        instance = object.__new__(cls)
        for name in (
            "semantic_view",
            "pixel_xy",
            "model_points",
            "source_vertex_indices",
            "source_weights",
            "confidence",
            "source_labels",
            "boundary_names",
            "depth",
        ):
            object.__setattr__(instance, name, getattr(validated, name))
        object.__setattr__(
            instance,
            "visible",
            _readonly_array(visibility, bool),
        )
        return instance


def _snapshot_region_masks(
    values: Mapping[str, np.ndarray],
    vertex_count: int,
) -> Mapping[str, np.ndarray]:
    if not isinstance(values, Mapping) or set(values) != set(_REGION_NAMES):
        raise ValueError(
            "semantic_region_masks must contain the canonical semantic regions"
        )
    snapshots = {}
    for name in _REGION_NAMES:
        mask = np.asarray(values[name])
        if mask.shape != (vertex_count,) or not np.issubdtype(
            mask.dtype,
            np.bool_,
        ):
            raise ValueError(
                f"semantic_region_masks[{name}] must be boolean shape (V,)"
            )
        snapshots[name] = _readonly_array(mask, bool)
    return MappingProxyType(snapshots)


def _validate_dynamic_projection(
    vertices: np.ndarray,
    views: Tuple[ProjectedNasalSamples, ...],
    target_names: Tuple[Tuple[str, ...], ...],
    region_masks: Mapping[str, np.ndarray],
) -> None:
    if (
        len(views) != len(NASAL_VIEWS)
        or not all(isinstance(value, ProjectedNasalSamples) for value in views)
        or tuple(value.semantic_view for value in views) != tuple(NASAL_VIEWS)
    ):
        raise ValueError("per_view samples must use canonical three-view order")
    if len(target_names) != len(NASAL_VIEWS) or any(
        not names or len(set(names)) != len(names)
        for names in target_names
    ):
        raise ValueError(
            "observation_target_names must contain non-empty unique "
            "targets for all three views"
        )
    for samples, allowed_targets in zip(views, target_names):
        if not len(samples.pixel_xy):
            raise ValueError(
                f"{samples.semantic_view} sample collection must not be empty"
            )
        if (
            np.any(samples.source_vertex_indices < 0)
            or np.any(samples.source_vertex_indices >= len(vertices))
        ):
            raise ValueError(
                f"{samples.semantic_view} source indices are out of range"
            )
        reconstructed = np.sum(
            vertices[samples.source_vertex_indices]
            * samples.source_weights[:, :, None],
            axis=1,
        )
        if not np.allclose(
            reconstructed,
            samples.model_points,
            atol=1e-10,
            rtol=1e-10,
        ):
            raise ValueError(
                f"{samples.semantic_view} provenance does not reconstruct "
                "model_points"
            )
        unknown_targets = set(samples.boundary_names) - set(allowed_targets)
        if unknown_targets:
            raise ValueError(
                f"{samples.semantic_view} boundary target is absent from "
                "the corresponding observation: "
                + ", ".join(sorted(unknown_targets))
            )
        for label, edge in zip(
            samples.source_labels,
            samples.source_vertex_indices,
        ):
            if not np.all(region_masks[label][edge]):
                raise ValueError(
                    f"{samples.semantic_view} source edge is not represented "
                    f"by semantic region {label}"
                )
    front_samples = views[0]
    for source_label, boundary_name in _FRONT_REGIONS:
        contributed = any(
            label == source_label and target == boundary_name
            for label, target in zip(
                front_samples.source_labels,
                front_samples.boundary_names,
            )
        )
        if not contributed:
            raise ValueError(
                f"mandatory front target {boundary_name} has no visible samples"
            )


@dataclass(frozen=True)
class MultiviewNasalProjection:
    """Candidate geometry and ordered immutable samples for all fixed views."""

    candidate_vertices: np.ndarray
    faces: np.ndarray
    per_view: Tuple[ProjectedNasalSamples, ...]
    observation_target_names: Tuple[Tuple[str, ...], ...]
    semantic_region_masks: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        vertices = _validate_vertices(self.candidate_vertices)
        faces = _validate_faces(self.faces, len(vertices))
        region_masks = _snapshot_region_masks(
            self.semantic_region_masks,
            len(vertices),
        )
        views = tuple(self.per_view)
        if (
            len(views) != len(NASAL_VIEWS)
            or not all(isinstance(value, ProjectedNasalSamples) for value in views)
            or tuple(value.semantic_view for value in views) != tuple(NASAL_VIEWS)
        ):
            raise ValueError("per_view samples must use canonical three-view order")
        target_names = tuple(
            tuple(str(name) for name in names)
            for names in self.observation_target_names
        )
        _validate_dynamic_projection(
            vertices,
            views,
            target_names,
            region_masks,
        )
        object.__setattr__(
            self,
            "candidate_vertices",
            _readonly_array(vertices, vertices.dtype),
        )
        object.__setattr__(self, "faces", _readonly_array(faces, faces.dtype))
        object.__setattr__(self, "per_view", views)
        object.__setattr__(
            self,
            "observation_target_names",
            target_names,
        )
        object.__setattr__(
            self,
            "semantic_region_masks",
            region_masks,
        )

    @classmethod
    def _from_prepared(
        cls,
        candidate: CandidateNasalMesh,
        per_view: Tuple[ProjectedNasalSamples, ...],
        prepared: PreparedNasalProjectionContext,
    ) -> "MultiviewNasalProjection":
        views = tuple(per_view)
        target_names = prepared.observation_target_names
        _validate_dynamic_projection(
            candidate.vertices,
            views,
            target_names,
            prepared.region_masks,
        )
        instance = object.__new__(cls)
        object.__setattr__(instance, "candidate_vertices", candidate.vertices)
        object.__setattr__(instance, "faces", prepared.faces)
        object.__setattr__(instance, "per_view", views)
        object.__setattr__(instance, "observation_target_names", target_names)
        object.__setattr__(
            instance,
            "semantic_region_masks",
            prepared.region_masks,
        )
        return instance

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


def _region_masks(
    semantic_basis,
    vertex_count: int,
    *,
    validate_semantic: bool = True,
) -> Mapping[str, np.ndarray]:
    if validate_semantic:
        _semantic_vectors(semantic_basis, vertex_count)
    raw = getattr(semantic_basis, "region_masks", None)
    if not isinstance(raw, Mapping):
        raise ValueError("semantic basis region_masks must be a mapping")
    result = {}
    support = np.asarray(semantic_basis.support_mask, dtype=bool)
    protected = np.asarray(semantic_basis.protected_mask, dtype=bool)
    if np.any(support & protected):
        raise ValueError("semantic support and protected masks must be disjoint")
    if set(raw) != set(_REGION_NAMES):
        raise ValueError(
            "semantic region masks must use the NasalSemanticBasis names"
        )
    for name in _REGION_NAMES:
        mask = np.asarray(raw[name])
        if mask.shape != (vertex_count,) or not np.issubdtype(
            mask.dtype, np.bool_
        ):
            raise ValueError(
                f"semantic region {name} must be boolean shape (V,)"
            )
        if np.any(mask & ~support) or np.any(mask & protected):
            raise ValueError(
                f"semantic region {name} extends outside editable support"
            )
        result[name] = np.asarray(mask, dtype=bool)
    return result


def _build_edge_adjacency_arrays(
    faces: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """Build lexicographically ordered manifold edge adjacency arrays."""
    face_count = len(faces)
    directed = np.vstack(
        (
            faces[:, (0, 1)],
            faces[:, (1, 2)],
            faces[:, (2, 0)],
        )
    ).astype(np.int64, copy=False)
    face_indices = np.tile(np.arange(face_count, dtype=np.int64), 3)
    undirected = np.sort(directed, axis=1)
    order = np.lexsort((undirected[:, 1], undirected[:, 0]))
    sorted_edges = undirected[order]
    sorted_faces = face_indices[order]
    starts = np.flatnonzero(
        np.concatenate(
            (
                np.ones(1, dtype=bool),
                np.any(sorted_edges[1:] != sorted_edges[:-1], axis=1),
            )
        )
    )
    ends = np.concatenate((starts[1:], np.array([len(sorted_edges)])))
    counts = ends - starts
    if np.any(counts > 2):
        raise ValueError("faces contain non-manifold topology edges")
    edge_vertices = sorted_edges[starts]
    edge_faces = np.full((len(starts), 2), -1, dtype=np.int64)
    edge_faces[:, 0] = sorted_faces[starts]
    paired = counts == 2
    edge_faces[paired, 1] = sorted_faces[starts[paired] + 1]
    return (
        _readonly_array(edge_vertices, np.int64),
        _readonly_array(edge_faces, np.int64),
    )


def _edge_adjacency(faces: np.ndarray):
    edge_vertices, edge_faces = _build_edge_adjacency_arrays(faces)
    return tuple(
        (
            int(edge[0]),
            int(edge[1]),
            tuple(int(value) for value in incident if value >= 0),
        )
        for edge, incident in zip(edge_vertices, edge_faces)
    )


def _closed_boundary_loops(
    edges: Sequence[Tuple[int, int]],
) -> Tuple[Tuple[int, ...], ...]:
    unvisited = {tuple(sorted(edge)) for edge in edges}
    neighbors = {}
    for first, second in unvisited:
        neighbors.setdefault(first, []).append(second)
        neighbors.setdefault(second, []).append(first)
    for values in neighbors.values():
        values.sort()
    loops = []
    while unvisited:
        first, second = min(unvisited)
        unvisited.remove((first, second))
        path = [first, second]
        previous = first
        current = second
        while current != path[0]:
            candidates = [
                neighbor
                for neighbor in neighbors.get(current, ())
                if tuple(sorted((current, neighbor))) in unvisited
            ]
            if not candidates:
                break
            nonbacktracking = [
                neighbor for neighbor in candidates if neighbor != previous
            ]
            next_vertex = min(nonbacktracking or candidates)
            unvisited.remove(tuple(sorted((current, next_vertex))))
            path.append(next_vertex)
            previous, current = current, next_vertex
        if path[-1] == path[0] and len(path) >= 4:
            loops.append(tuple(path[:-1]))
    return tuple(loops)


def _projected_loop_area(
    loop: Sequence[int],
    projected_vertices: np.ndarray,
) -> float:
    points = projected_vertices[np.asarray(loop, dtype=np.int64)]
    next_points = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(points[:, 0] * next_points[:, 1])
        - np.sum(points[:, 1] * next_points[:, 0])
    )


def _prepare_front_composite_loops(
    faces: np.ndarray,
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    regions: Mapping[str, np.ndarray],
    protected_mask: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    composite = (
        regions["subject_left_nose_wing"]
        | regions["subject_right_nose_wing"]
        | regions["nose_tip"]
        | regions["tip_alar_transition"]
    ) & ~np.asarray(protected_mask, dtype=bool)
    active_faces = np.all(composite[faces], axis=1)
    incident_active = active_faces[edge_faces[:, 0]].astype(np.int8)
    paired = edge_faces[:, 1] >= 0
    incident_active[paired] += active_faces[
        edge_faces[paired, 1]
    ].astype(np.int8)
    boundary_edges = edge_vertices[incident_active == 1]
    loops = _closed_boundary_loops(
        tuple((int(edge[0]), int(edge[1])) for edge in boundary_edges)
    )
    if not loops:
        raise ValueError("no valid external composite nasal contour exists")

    flattened = []
    offsets = [0]
    source_codes = []
    for loop in loops:
        flattened.extend(loop)
        for index, first in enumerate(loop):
            second = loop[(index + 1) % len(loop)]
            represented = [
                region_index
                for region_index, (name, _target) in enumerate(_FRONT_REGIONS)
                if regions[name][first] and regions[name][second]
            ]
            source_codes.append(represented[0] if len(represented) == 1 else -1)
        offsets.append(len(flattened))
    return (
        _readonly_array(np.asarray(flattened, dtype=np.int64), np.int64),
        _readonly_array(np.asarray(offsets, dtype=np.int64), np.int64),
        _readonly_array(np.asarray(source_codes, dtype=np.int8), np.int8),
    )


def _prepared_front_edge_groups(
    prepared: PreparedNasalProjectionContext,
    projected_vertices: np.ndarray,
):
    loops = tuple(
        tuple(
            int(value)
            for value in prepared.front_loop_vertices[
                prepared.front_loop_offsets[index] :
                prepared.front_loop_offsets[index + 1]
            ]
        )
        for index in range(len(prepared.front_loop_offsets) - 1)
    )
    ranked = sorted(
        range(len(loops)),
        key=lambda index: (
            -abs(_projected_loop_area(loops[index], projected_vertices)),
            loops[index],
        ),
    )
    loop_index = ranked[0]
    loop = loops[loop_index]
    start = int(prepared.front_loop_offsets[loop_index])
    codes = prepared.front_loop_source_codes[start : start + len(loop)]
    classified = {index: [] for index in range(len(_FRONT_REGIONS))}
    for index, code in enumerate(codes):
        if int(code) >= 0:
            edge = tuple(
                sorted((loop[index], loop[(index + 1) % len(loop)]))
            )
            classified[int(code)].append(edge)
    return tuple(
        (
            source_label,
            boundary_name,
            tuple(sorted(classified[index])),
        )
        for index, (source_label, boundary_name) in enumerate(_FRONT_REGIONS)
    )


def _external_region_boundary_edges(
    adjacency,
    faces: np.ndarray,
    region_mask: np.ndarray,
    protected_mask: np.ndarray,
    projected_vertices: np.ndarray,
) -> Tuple[Tuple[int, int], ...]:
    editable_region = np.asarray(region_mask, dtype=bool) & ~np.asarray(
        protected_mask,
        dtype=bool,
    )
    active_faces = np.all(editable_region[faces], axis=1)
    boundary_edges = []
    for first, second, incident in adjacency:
        incident_active = sum(bool(active_faces[index]) for index in incident)
        if incident_active == 1:
            boundary_edges.append((first, second))
    loops = _closed_boundary_loops(boundary_edges)
    if not loops:
        return ()
    ranked = sorted(
        loops,
        key=lambda loop: (
            -abs(_projected_loop_area(loop, projected_vertices)),
            tuple(loop),
        ),
    )
    external = ranked[0]
    return tuple(
        sorted(
            tuple(sorted((external[index], external[(index + 1) % len(external)])))
            for index in range(len(external))
        )
    )


def _front_external_edge_groups(
    adjacency,
    faces: np.ndarray,
    regions: Mapping[str, np.ndarray],
    protected_mask: np.ndarray,
    projected_vertices: np.ndarray,
):
    composite = (
        regions["subject_left_nose_wing"]
        | regions["subject_right_nose_wing"]
        | regions["nose_tip"]
        | regions["tip_alar_transition"]
    ) & ~np.asarray(protected_mask, dtype=bool)
    external_edges = _external_region_boundary_edges(
        adjacency,
        faces,
        composite,
        protected_mask,
        projected_vertices,
    )
    classified = {name: [] for name, _target in _FRONT_REGIONS}
    for edge in external_edges:
        represented = [
            name
            for name, _target in _FRONT_REGIONS
            if np.all(regions[name][np.asarray(edge, dtype=np.int64)])
        ]
        if len(represented) == 1:
            classified[represented[0]].append(edge)
    return tuple(
        (
            source_label,
            boundary_name,
            tuple(sorted(classified[source_label])),
        )
        for source_label, boundary_name in _FRONT_REGIONS
    )


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
            if nondegenerate[incident[0]] and front_facing[incident[0]]:
                result.append((first, second))
        elif (
            nondegenerate[incident[0]]
            and nondegenerate[incident[1]]
            and front_facing[incident[0]] != front_facing[incident[1]]
        ):
            result.append((first, second))
    return tuple(result)


def _silhouette_edges_from_arrays(
    edge_vertices: np.ndarray,
    edge_faces: np.ndarray,
    camera_points: np.ndarray,
    faces: np.ndarray,
) -> np.ndarray:
    triangles = camera_points[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    centroids = np.mean(triangles, axis=1)
    facing_measure = np.einsum("fi,fi->f", normals, -centroids)
    nondegenerate = np.linalg.norm(normals, axis=1) > 1e-12
    front_facing = facing_measure > 0.0
    first_faces = edge_faces[:, 0]
    second_faces = edge_faces[:, 1]
    boundary = second_faces < 0
    keep = boundary & nondegenerate[first_faces] & front_facing[first_faces]
    paired = ~boundary
    paired_indices = np.flatnonzero(paired)
    if len(paired_indices):
        first = first_faces[paired_indices]
        second = second_faces[paired_indices]
        keep[paired_indices] = (
            nondegenerate[first]
            & nondegenerate[second]
            & (front_facing[first] != front_facing[second])
        )
    return edge_vertices[keep]


def _sample_edges(
    edges: Sequence[Tuple[int, int]],
    count: int,
    vertices: np.ndarray,
    projected_vertices: np.ndarray,
    camera_depth: np.ndarray,
):
    depth = np.asarray(camera_depth, dtype=np.float64)
    if depth.shape != (len(vertices),) or not np.isfinite(depth).all():
        raise ValueError("camera endpoint depths must be finite shape (V,)")
    usable = []
    for first, second in sorted(edges):
        length = float(
            np.linalg.norm(projected_vertices[second] - projected_vertices[first])
        )
        if (
            np.isfinite(length)
            and length > 1e-10
            and depth[first] > 0.0
            and depth[second] > 0.0
        ):
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
        screen_fraction = float(
            (target - starts[int(segment_index)]) / length
        )
        denominator = (
            (1.0 - screen_fraction) * depth[second]
            + screen_fraction * depth[first]
        )
        if not np.isfinite(denominator) or denominator <= 0.0:
            raise ValueError(
                "perspective edge interpolation has invalid depth denominator"
            )
        alpha = float(screen_fraction * depth[first] / denominator)
        if not np.isfinite(alpha) or alpha < 0.0 or alpha > 1.0:
            raise ValueError("perspective edge interpolation is invalid")
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
) -> Optional[str]:
    for name in _SOURCE_PRIORITY:
        if regions[name][first] and regions[name][second]:
            return name
    return None


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
    regions: Optional[Mapping[str, np.ndarray]] = None,
    vertex_projection=None,
) -> ProjectedNasalSamples:
    if vertex_projection is None:
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
            vertex_projection.depth,
        )
        all_points.append(points)
        all_indices.append(indices)
        all_weights.append(weights)
        if source_label is None:
            if regions is None:
                raise ValueError("semantic regions are required for edge labeling")
            labels = [
                _source_label(int(edge[0]), int(edge[1]), regions)
                for edge in indices
            ]
            if any(label is None for label in labels):
                raise ValueError(
                    "profile edge has no common canonical semantic source"
                )
            all_labels.extend(labels)
        else:
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
    def metadata_array(metadata, key, shape, label):
        if key not in metadata:
            raise ValueError(f"{label} is missing {key}")
        try:
            value = np.asarray(metadata[key], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{label} {key} must be numeric") from exc
        if value.shape != shape or not np.isfinite(value).all():
            raise ValueError(f"{label} {key} must have finite shape {shape}")
        return value

    for view, semantic_view in zip(views, NASAL_VIEWS):
        observation = observations.by_view[semantic_view]
        camera = observation.camera
        camera_metadata = observation.camera_metadata
        coordinate_metadata = observation.coordinate_metadata
        if tuple(observation.original_size) != tuple(camera.image_size):
            raise ValueError(
                f"{semantic_view} observation original size disagrees with camera"
            )
        expected_camera_scalars = {
            "camera_name": camera.name,
            "camera_view": camera.view,
            "subject_relative_view": semantic_view,
            "image_size_wh": tuple(camera.image_size),
        }
        for key, expected_value in expected_camera_scalars.items():
            if key not in camera_metadata:
                raise ValueError(f"{semantic_view} camera_metadata is missing {key}")
            actual_value = camera_metadata[key]
            if key == "image_size_wh":
                actual_value = tuple(int(value) for value in actual_value)
            if actual_value != expected_value:
                raise ValueError(
                    f"{semantic_view} camera_metadata does not match camera {key}"
                )
        camera_arrays = (
            ("intrinsics", camera.K, (3, 3)),
            (
                "distortion_coefficients",
                np.asarray(camera.dist).reshape(-1),
                np.asarray(camera.dist).reshape(-1).shape,
            ),
            ("rig_to_camera_rotation", camera.R_rig_to_camera, (3, 3)),
            ("rig_to_camera_translation", camera.t_rig_to_camera, (3,)),
        )
        for key, expected_value, shape in camera_arrays:
            actual = metadata_array(
                camera_metadata,
                key,
                shape,
                f"{semantic_view} camera_metadata",
            )
            if not np.allclose(
                actual,
                np.asarray(expected_value, dtype=np.float64).reshape(shape),
                atol=1e-12,
                rtol=0.0,
            ):
                raise ValueError(
                    f"{semantic_view} camera_metadata does not match camera {key}"
                )
        required_coordinate_scalars = {
            "original_size_wh": tuple(observation.original_size),
            "work_size_wh": tuple(observation.work_size),
            "observation_pixel_frame": "undistorted_work_px",
        }
        for key, expected_value in required_coordinate_scalars.items():
            if key not in coordinate_metadata:
                raise ValueError(
                    f"{semantic_view} coordinate_metadata is missing {key}"
                )
            actual_value = coordinate_metadata[key]
            if key.endswith("_size_wh"):
                actual_value = tuple(int(value) for value in actual_value)
            if actual_value != expected_value:
                raise ValueError(
                    f"{semantic_view} coordinate metadata {key} is "
                    "inconsistent with the observation"
                )
        coordinate_K = metadata_array(
            coordinate_metadata,
            "intrinsics",
            (3, 3),
            f"{semantic_view} coordinate_metadata",
        )
        expected = scale_intrinsics(
            coordinate_K,
            tuple(observation.original_size),
            tuple(observation.work_size),
        )
        if not np.allclose(view.K, expected, atol=1e-9, rtol=1e-9):
            raise ValueError(
                f"{semantic_view} ProjectionView K does not match the "
                "authoritative coordinate work-frame contract"
            )


def prepare_nasal_projection_context(
    faces: np.ndarray,
    semantic_basis,
    observations: NasalObservationBundle,
    views: Sequence[ProjectionView],
    *,
    config: Optional[MultiviewNasalSamplingConfig] = None,
) -> PreparedNasalProjectionContext:
    """Validate and snapshot all candidate-independent projection state."""
    if not isinstance(observations, NasalObservationBundle):
        raise ValueError("observations must be a NasalObservationBundle")
    limits = MultiviewNasalSamplingConfig() if config is None else config
    if not isinstance(limits, MultiviewNasalSamplingConfig):
        raise ValueError("config must be a MultiviewNasalSamplingConfig")
    canonical_views = _canonical_views(views)
    _validate_work_intrinsics(observations, canonical_views)

    support = np.asarray(getattr(semantic_basis, "support_mask", None))
    if support.ndim != 1 or len(support) < 3 or not np.issubdtype(
        support.dtype,
        np.bool_,
    ):
        raise ValueError("semantic basis support_mask must be boolean shape (V,)")
    vertex_count = len(support)
    semantic_vectors = _semantic_vectors(semantic_basis, vertex_count)
    regions = _region_masks(
        semantic_basis,
        vertex_count,
        validate_semantic=False,
    )
    protected = np.asarray(semantic_basis.protected_mask, dtype=bool)

    view_snapshots = tuple(
        ProjectionView(
            view.name,
            view.K,
            view.R_model_to_camera,
            view.t_model_to_camera,
        )
        for view in canonical_views
    )
    contracts = []
    for semantic_view in NASAL_VIEWS:
        observation = observations.by_view[semantic_view]
        target_names = tuple(sorted(str(name) for name in observation.distance_fields))
        mandatory_targets = (
            {"subject-left-alar", "subject-right-alar"}
            if semantic_view == "front"
            else {"nasal-profile"}
        )
        missing = mandatory_targets - set(target_names)
        if missing:
            raise ValueError(
                f"{semantic_view} observation is missing mandatory targets: "
                + ", ".join(sorted(missing))
            )
        contracts.append(
            _PreparedViewTargetContract(
                semantic_view=semantic_view,
                work_size=tuple(observation.work_size),
                roi_work_xyxy=tuple(observation.roi_work_xyxy),
                confidence=observation.confidence,
                target_names=target_names,
            )
        )
    limits_snapshot = MultiviewNasalSamplingConfig(
        front_samples_per_region=int(limits.front_samples_per_region),
        side_samples_per_view=int(limits.side_samples_per_view),
        min_depth=float(limits.min_depth),
        visibility_relative_tolerance=float(
            limits.visibility_relative_tolerance
        ),
        visibility_absolute_tolerance=float(
            limits.visibility_absolute_tolerance
        ),
        barycentric_tolerance=float(limits.barycentric_tolerance),
    )
    return PreparedNasalProjectionContext(
        vertex_count=vertex_count,
        faces=faces,
        semantic_vectors=semantic_vectors,
        support_mask=support,
        protected_mask=protected,
        region_masks=regions,
        views=view_snapshots,
        view_contracts=tuple(contracts),
        config=limits_snapshot,
    )


def _current_prepared_edge_groups(
    semantic_view: str,
    prepared: PreparedNasalProjectionContext,
    vertex_projection,
):
    """Return current C1 front exterior or view-dependent silhouette edges."""
    regions = prepared.region_masks
    limits = prepared.config
    if semantic_view == "front":
        front_groups = _prepared_front_edge_groups(
            prepared,
            vertex_projection.pixel_xy,
        )
        return [
            (
                edges,
                int(limits.front_samples_per_region),
                source_label,
                boundary_name,
            )
            for source_label, boundary_name, edges in front_groups
        ]
    silhouettes = _silhouette_edges_from_arrays(
        prepared.edge_vertices,
        prepared.edge_faces,
        vertex_projection.camera_points,
        prepared.faces,
    )
    nasal_profile = (
        prepared.support_mask
        & (
            regions["nose_tip"]
            | regions["tip_alar_transition"]
            | regions["subject_left_nose_wing"]
            | regions["subject_right_nose_wing"]
        )
    )
    restricted = tuple(
        (int(edge[0]), int(edge[1]))
        for edge in silhouettes
        if nasal_profile[edge[0]] and nasal_profile[edge[1]]
    )
    by_label = {}
    for edge in restricted:
        label = _source_label(edge[0], edge[1], regions)
        if label is None:
            continue
        by_label.setdefault(label, []).append(edge)
    ordered_edges = tuple(
        edge
        for label in _SOURCE_PRIORITY
        for edge in sorted(by_label.get(label, ()))
    )
    return [
        (
            ordered_edges,
            int(limits.side_samples_per_view),
            None,
            "nasal-profile",
        )
    ]


def project_multiview_nasal_boundaries_prepared(
    candidate: CandidateNasalMesh,
    prepared: PreparedNasalProjectionContext,
) -> MultiviewNasalProjection:
    """Project one candidate using previously validated static context."""
    if not isinstance(candidate, CandidateNasalMesh):
        raise ValueError("candidate must be a CandidateNasalMesh")
    if not isinstance(prepared, PreparedNasalProjectionContext):
        raise ValueError("prepared must be a PreparedNasalProjectionContext")
    if len(candidate.vertices) != prepared.vertex_count:
        raise ValueError("candidate vertex count does not match prepared context")
    if candidate.faces is not prepared.faces:
        raise ValueError(
            "prepared projection requires candidate faces from the same context"
        )

    per_view = []
    regions = prepared.region_masks
    limits = prepared.config
    for semantic_view, view, observation in zip(
        NASAL_VIEWS,
        prepared.views,
        prepared.view_contracts,
    ):
        vertex_projection = project_points_strict(
            candidate.vertices,
            view.K,
            view.R_model_to_camera,
            view.t_model_to_camera,
            epsilon=float(limits.min_depth),
        )
        edge_groups = _current_prepared_edge_groups(
            semantic_view,
            prepared,
            vertex_projection,
        )
        samples = _project_edge_groups(
            semantic_view,
            edge_groups,
            candidate,
            observation,
            view,
            limits,
            regions,
            vertex_projection,
        )
        per_view.append(samples)
    return MultiviewNasalProjection._from_prepared(
        candidate,
        tuple(per_view),
        prepared,
    )


def project_multiview_nasal_boundaries(
    candidate: CandidateNasalMesh,
    semantic_basis,
    observations: NasalObservationBundle,
    views: Sequence[ProjectionView],
    *,
    config: Optional[MultiviewNasalSamplingConfig] = None,
) -> MultiviewNasalProjection:
    """Prepare once and project semantic front and visible side boundaries."""
    if not isinstance(candidate, CandidateNasalMesh):
        raise ValueError("candidate must be a CandidateNasalMesh")
    prepared = prepare_nasal_projection_context(
        candidate.faces,
        semantic_basis,
        observations,
        views,
        config=config,
    )
    prepared_candidate = CandidateNasalMesh._from_validated(
        candidate.vertices,
        prepared.faces,
        reuse_faces=True,
        reuse_vertices=True,
    )
    return project_multiview_nasal_boundaries_prepared(
        prepared_candidate,
        prepared,
    )


_OBJECTIVE_IMAGE_TERM_SPECS = (
    (
        "front_subject_left_alar",
        "front",
        "subject-left-alar",
    ),
    (
        "front_subject_right_alar",
        "front",
        "subject-right-alar",
    ),
    (
        "subject_left_nasal_profile",
        "subject-left",
        "nasal-profile",
    ),
    (
        "subject_right_nasal_profile",
        "subject-right",
        "nasal-profile",
    ),
)
_OBJECTIVE_IMAGE_TERM_NAMES = tuple(
    value[0] for value in _OBJECTIVE_IMAGE_TERM_SPECS
)
_OBJECTIVE_REGULARIZATION_TERM_NAMES = (
    "projection_depth_barrier",
    "flame_prior",
    "semantic_prior",
    "surface_smoothness",
    "surface_orientation_barrier",
    "weak_symmetry",
)
_ASYMMETRY_MODE_NAMES = (
    "alar_width_asymmetry",
    "alar_depth_asymmetry",
)
_MIN_DEPTH_BARRIER_WEIGHT = 100.0
_FIXED_DEPTH_BARRIER_SCALE = 0.02
_FIXED_DEPTH_BARRIER_SOFTMIN_TEMPERATURE = 1e-4
_FIXED_ORIENTATION_BARRIER_WEIGHT = 25.0
_FIXED_ORIENTATION_BARRIER_MARGIN = 0.20
_FIXED_ORIENTATION_BARRIER_SCALE = 0.05
_FIXED_ORIENTATION_BARRIER_SOFTMIN_TEMPERATURE = 0.01
_ORIENTATION_BASELINE_LOCAL_QUALITY_THRESHOLD = 1e-6
_ORIENTATION_DEGENERATE_FACE_PREVIEW_COUNT = 20
_MIN_ROBUST_F_SCALE = 1.0
_FIXED_OBSERVABLE_COEFFICIENT_BOUND = 3.0
_FIXED_SEMANTIC_COEFFICIENT_BOUND = 3.0


def _positive_tuple(
    name: str,
    values,
    length: int,
) -> Tuple[float, ...]:
    try:
        result = tuple(float(value) for value in values)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"{name} must contain {length} finite positive values"
        ) from exc
    if (
        len(result) != length
        or not np.isfinite(np.asarray(result, dtype=np.float64)).all()
        or any(value <= 0.0 for value in result)
    ):
        raise ValueError(
            f"{name} must contain {length} finite positive values"
        )
    return result


@dataclass(frozen=True)
class MultiviewNasalObjectiveConfig:
    """Frozen subject-independent residual, visibility, and solver scales.

    Profile spatial scales use work pixels; all depth scales use fixed camera
    depth units. Coefficient bounds and feasibility transition scales cannot
    be varied per dataset.
    """

    front_image_weight: float = 1.0
    side_image_weight: float = 1.0
    flame_prior_weight: float = 0.15
    semantic_prior_weight: float = 0.10
    semantic_prior_standard_deviations: Tuple[float, ...] = (
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
        1.0,
    )
    smoothness_weight: float = 0.10
    smoothness_scale: float = 1.0
    symmetry_weight: float = 0.10
    symmetry_evidence_floor: float = 0.15
    symmetry_evidence_ceiling: float = 1.0
    profile_vertical_sigma_px: float = 6.0
    profile_softmax_temperature_px: float = 20.0
    profile_visibility_spatial_sigma_px: float = 64.0
    profile_front_depth_temperature: float = 0.05
    profile_depth_visibility_scale: float = 2.0
    profile_depth_validity_scale: float = 0.02
    depth_softplus_scale: float = 0.02
    depth_barrier_weight: float = _MIN_DEPTH_BARRIER_WEIGHT
    depth_barrier_scale: float = _FIXED_DEPTH_BARRIER_SCALE
    depth_barrier_softmin_temperature: float = (
        _FIXED_DEPTH_BARRIER_SOFTMIN_TEMPERATURE
    )
    orientation_barrier_weight: float = (
        _FIXED_ORIENTATION_BARRIER_WEIGHT
    )
    orientation_barrier_margin: float = (
        _FIXED_ORIENTATION_BARRIER_MARGIN
    )
    orientation_barrier_scale: float = (
        _FIXED_ORIENTATION_BARRIER_SCALE
    )
    orientation_barrier_softmin_temperature: float = (
        _FIXED_ORIENTATION_BARRIER_SOFTMIN_TEMPERATURE
    )
    observable_coefficient_bound: float = (
        _FIXED_OBSERVABLE_COEFFICIENT_BOUND
    )
    semantic_coefficient_bound: float = _FIXED_SEMANTIC_COEFFICIENT_BOUND
    robust_loss: str = "soft_l1"
    robust_f_scale: float = 1.0

    def __post_init__(self) -> None:
        for name in (
            "front_image_weight",
            "side_image_weight",
            "flame_prior_weight",
            "semantic_prior_weight",
            "smoothness_weight",
            "symmetry_weight",
            "depth_barrier_weight",
            "orientation_barrier_weight",
        ):
            _finite_real(name, getattr(self, name))
        if float(self.depth_barrier_weight) < _MIN_DEPTH_BARRIER_WEIGHT:
            raise ValueError(
                "depth_barrier_weight must be at least "
                f"{_MIN_DEPTH_BARRIER_WEIGHT:g}"
            )
        for name in (
            "smoothness_scale",
            "profile_vertical_sigma_px",
            "profile_softmax_temperature_px",
            "profile_visibility_spatial_sigma_px",
            "profile_front_depth_temperature",
            "profile_depth_visibility_scale",
            "profile_depth_validity_scale",
            "depth_softplus_scale",
            "depth_barrier_scale",
            "depth_barrier_softmin_temperature",
            "orientation_barrier_margin",
            "orientation_barrier_scale",
            "orientation_barrier_softmin_temperature",
            "robust_f_scale",
        ):
            _finite_real(name, getattr(self, name), strictly_positive=True)
        if float(self.depth_barrier_scale) != _FIXED_DEPTH_BARRIER_SCALE:
            raise ValueError(
                "depth_barrier_scale is the fixed feasibility value "
                f"{_FIXED_DEPTH_BARRIER_SCALE:g}"
            )
        if (
            float(self.depth_barrier_softmin_temperature)
            != _FIXED_DEPTH_BARRIER_SOFTMIN_TEMPERATURE
        ):
            raise ValueError(
                "depth_barrier_softmin_temperature is the fixed "
                "feasibility value "
                f"{_FIXED_DEPTH_BARRIER_SOFTMIN_TEMPERATURE:g}"
            )
        for name, expected in (
            (
                "orientation_barrier_weight",
                _FIXED_ORIENTATION_BARRIER_WEIGHT,
            ),
            (
                "orientation_barrier_margin",
                _FIXED_ORIENTATION_BARRIER_MARGIN,
            ),
            (
                "orientation_barrier_scale",
                _FIXED_ORIENTATION_BARRIER_SCALE,
            ),
            (
                "orientation_barrier_softmin_temperature",
                _FIXED_ORIENTATION_BARRIER_SOFTMIN_TEMPERATURE,
            ),
        ):
            if float(getattr(self, name)) != expected:
                raise ValueError(
                    f"{name} is the fixed subject-independent value "
                    f"{expected:g}"
                )
        if float(self.robust_f_scale) < _MIN_ROBUST_F_SCALE:
            raise ValueError(
                f"robust_f_scale must be at least {_MIN_ROBUST_F_SCALE:g}"
            )
        for name, expected in (
            (
                "observable_coefficient_bound",
                _FIXED_OBSERVABLE_COEFFICIENT_BOUND,
            ),
            (
                "semantic_coefficient_bound",
                _FIXED_SEMANTIC_COEFFICIENT_BOUND,
            ),
        ):
            value = _finite_real(
                name,
                getattr(self, name),
                strictly_positive=True,
            )
            if value != expected:
                raise ValueError(
                    f"{name} is the fixed subject-independent value "
                    f"{expected:g}"
                )
        floor = _finite_real(
            "symmetry_evidence_floor",
            self.symmetry_evidence_floor,
            strictly_positive=True,
        )
        ceiling = _finite_real(
            "symmetry_evidence_ceiling",
            self.symmetry_evidence_ceiling,
            strictly_positive=True,
        )
        if ceiling < floor:
            raise ValueError(
                "symmetry_evidence_ceiling must be at least "
                "symmetry_evidence_floor"
            )
        if str(self.robust_loss) != "soft_l1":
            raise ValueError("robust_loss must be the fixed value 'soft_l1'")
        semantic_scales = _positive_tuple(
            "semantic_prior_standard_deviations",
            self.semantic_prior_standard_deviations,
            len(NASAL_SEMANTIC_MODE_NAMES),
        )
        object.__setattr__(
            self,
            "semantic_prior_standard_deviations",
            semantic_scales,
        )
        object.__setattr__(self, "robust_loss", "soft_l1")


@dataclass(frozen=True)
class _PreparedObjectiveImageTerm:
    name: str
    semantic_view: str
    boundary_name: str
    slot_quantiles: np.ndarray
    target_slot_xy: np.ndarray
    target_polyline: np.ndarray
    support_vertex_indices: np.ndarray
    direction_sign: int
    distance_field: np.ndarray
    confidence: np.ndarray
    work_size: Tuple[int, int]
    roi_work_xyxy: Tuple[float, float, float, float]
    target_segment_starts: np.ndarray = field(init=False)
    target_segment_deltas: np.ndarray = field(init=False)
    target_segment_length_squared: np.ndarray = field(init=False)
    slot_y: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        valid_specs = {
            name: (semantic_view, boundary_name)
            for name, semantic_view, boundary_name
            in _OBJECTIVE_IMAGE_TERM_SPECS
        }
        name = str(self.name)
        semantic_view = str(self.semantic_view)
        boundary_name = str(self.boundary_name)
        if (
            name not in valid_specs
            or valid_specs[name] != (semantic_view, boundary_name)
        ):
            raise ValueError("objective image term name/view/target is invalid")
        quantiles = np.asarray(self.slot_quantiles, dtype=np.float64)
        if (
            quantiles.ndim != 1
            or not len(quantiles)
            or not np.isfinite(quantiles).all()
            or np.any((quantiles <= 0.0) | (quantiles >= 1.0))
        ):
            raise ValueError(
                "objective slot quantiles must be a finite vector in (0, 1)"
            )
        expected_quantiles = (
            np.arange(len(quantiles), dtype=np.float64) + 0.5
        ) / float(len(quantiles))
        if not np.array_equal(quantiles, expected_quantiles):
            raise ValueError(
                "objective slots must use deterministic midpoint quantiles"
            )
        target_slots = np.asarray(self.target_slot_xy, dtype=np.float64)
        polyline = np.asarray(self.target_polyline, dtype=np.float64)
        support_indices = np.asarray(self.support_vertex_indices)
        if (
            target_slots.shape != (len(quantiles), 2)
            or not np.isfinite(target_slots).all()
        ):
            raise ValueError(
                "objective target slots must have finite shape (slot_count, 2)"
            )
        if (
            polyline.ndim != 2
            or polyline.shape[1:] != (2,)
            or not len(polyline)
            or not np.isfinite(polyline).all()
        ):
            raise ValueError(
                "objective target polyline must have finite shape (N, 2)"
            )
        if (
            support_indices.ndim != 1
            or not len(support_indices)
            or not np.issubdtype(support_indices.dtype, np.integer)
            or np.any(support_indices < 0)
            or np.any(np.diff(support_indices) <= 0)
        ):
            raise ValueError(
                "objective semantic support indices must be sorted unique "
                "non-negative integers"
            )
        direction = int(self.direction_sign)
        if (
            isinstance(self.direction_sign, (bool, np.bool_))
            or direction not in (-1, 1)
        ):
            raise ValueError("objective profile direction_sign must be -1 or 1")
        distance = np.asarray(self.distance_field, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        work_size = tuple(int(value) for value in self.work_size)
        roi = tuple(float(value) for value in self.roi_work_xyxy)
        if (
            len(work_size) != 2
            or min(work_size) < 1
            or distance.shape != (work_size[1], work_size[0])
            or confidence.shape != quantiles.shape
        ):
            raise ValueError(
                "objective image fields and per-slot confidence have "
                "inconsistent shapes"
            )
        if (
            not np.isfinite(distance).all()
            or np.any(distance < 0.0)
        ):
            raise ValueError(
                "objective distance fields must contain finite unsigned "
                "pixel distances"
            )
        if (
            not np.isfinite(confidence).all()
            or np.any((confidence < 0.0) | (confidence > 1.0))
        ):
            raise ValueError(
                "objective confidence fields must lie in [0, 1]"
            )
        if (
            len(roi) != 4
            or not np.isfinite(np.asarray(roi)).all()
            or not (
                0.0 <= roi[0] < roi[2] <= float(work_size[0])
                and 0.0 <= roi[1] < roi[3] <= float(work_size[1])
            )
        ):
            raise ValueError("objective image ROI is invalid")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "semantic_view", semantic_view)
        object.__setattr__(self, "boundary_name", boundary_name)
        object.__setattr__(
            self,
            "slot_quantiles",
            _readonly_array(quantiles, np.float64),
        )
        object.__setattr__(
            self,
            "target_slot_xy",
            _readonly_array(target_slots, np.float64),
        )
        object.__setattr__(
            self,
            "target_polyline",
            _readonly_array(polyline, np.float64),
        )
        object.__setattr__(
            self,
            "support_vertex_indices",
            _readonly_array(support_indices, np.int64),
        )
        object.__setattr__(self, "direction_sign", direction)
        object.__setattr__(
            self,
            "distance_field",
            _readonly_array(distance, np.float64),
        )
        object.__setattr__(
            self,
            "confidence",
            _readonly_array(confidence, np.float64),
        )
        object.__setattr__(self, "work_size", work_size)
        object.__setattr__(self, "roi_work_xyxy", roi)
        if len(polyline) == 1:
            segment_starts = polyline
            segment_deltas = np.zeros((1, 2), dtype=np.float64)
        else:
            raw_deltas = np.diff(polyline, axis=0)
            usable = np.sum(raw_deltas * raw_deltas, axis=1) > 1e-20
            if np.any(usable):
                segment_starts = polyline[:-1][usable]
                segment_deltas = raw_deltas[usable]
            else:
                segment_starts = polyline[:1]
                segment_deltas = np.zeros((1, 2), dtype=np.float64)
        segment_length_squared = np.sum(
            segment_deltas * segment_deltas,
            axis=1,
        )
        object.__setattr__(
            self,
            "target_segment_starts",
            _readonly_array(segment_starts, np.float64),
        )
        object.__setattr__(
            self,
            "target_segment_deltas",
            _readonly_array(segment_deltas, np.float64),
        )
        object.__setattr__(
            self,
            "target_segment_length_squared",
            _readonly_array(segment_length_squared, np.float64),
        )
        object.__setattr__(
            self,
            "slot_y",
            _readonly_array(target_slots[:, 1], np.float64),
        )

    @property
    def sample_count(self) -> int:
        return len(self.slot_quantiles)


@dataclass(frozen=True)
class MultiviewNasalObjectiveContext:
    """Static, deeply immutable state reused by objective evaluations."""

    baseline_vertices: np.ndarray
    observable_vertex_basis: np.ndarray
    projection_context: PreparedNasalProjectionContext
    image_terms: Tuple[_PreparedObjectiveImageTerm, ...]
    depth_support_vertex_indices: np.ndarray
    flame_mode_standard_deviations: np.ndarray
    smoothness_centers: np.ndarray
    smoothness_neighbors: np.ndarray
    smoothness_neighbor_offsets: np.ndarray
    smoothness_edge_count: int
    orientation_face_indices: np.ndarray
    orientation_face_vertices: np.ndarray
    orientation_reference_cross: np.ndarray
    orientation_reference_inverse_squared_norm: np.ndarray
    parameter_ordering: Tuple[str, ...]
    parameter_lower_bounds: np.ndarray
    parameter_upper_bounds: np.ndarray

    def __post_init__(self) -> None:
        if not isinstance(
            self.projection_context,
            PreparedNasalProjectionContext,
        ):
            raise ValueError(
                "projection_context must be a PreparedNasalProjectionContext"
            )
        baseline = _validate_vertices(self.baseline_vertices)
        prepared = self.projection_context
        if len(baseline) != prepared.vertex_count:
            raise ValueError(
                "baseline vertex count does not match projection context"
            )
        flame_basis = _validate_basis_array(
            "observable vertex basis",
            self.observable_vertex_basis,
            (prepared.vertex_count, 3, None),
        )
        rank = flame_basis.shape[2]
        standard_deviations = np.asarray(
            self.flame_mode_standard_deviations,
            dtype=np.float64,
        )
        if (
            standard_deviations.shape != (rank,)
            or not np.isfinite(standard_deviations).all()
            or np.any(standard_deviations <= 0.0)
        ):
            raise ValueError(
                "flame mode standard deviations must be finite positive "
                "shape (rank,)"
            )
        terms = tuple(self.image_terms)
        if (
            len(terms) != len(_OBJECTIVE_IMAGE_TERM_SPECS)
            or not all(
                isinstance(term, _PreparedObjectiveImageTerm)
                for term in terms
            )
            or tuple(term.name for term in terms)
            != _OBJECTIVE_IMAGE_TERM_NAMES
        ):
            raise ValueError(
                "objective image terms must use the fixed canonical order"
            )
        for term in terms:
            expected_count = (
                int(prepared.config.front_samples_per_region)
                if term.semantic_view == "front"
                else int(prepared.config.side_samples_per_view)
            )
            contract = prepared.view_contracts[
                NASAL_VIEWS.index(term.semantic_view)
            ]
            if term.sample_count != expected_count:
                raise ValueError(
                    f"objective image term {term.name} has the wrong "
                    "canonical slot count"
                )
            if (
                term.work_size != contract.work_size
                or term.roi_work_xyxy != contract.roi_work_xyxy
                or term.boundary_name not in contract.target_names
            ):
                raise ValueError(
                    f"objective image term {term.name} does not match "
                    "its prepared observation contract"
                )
            if (
                np.any(term.support_vertex_indices >= prepared.vertex_count)
                or np.any(prepared.protected_mask[term.support_vertex_indices])
                or np.any(~prepared.support_mask[term.support_vertex_indices])
            ):
                raise ValueError(
                    f"objective semantic support is invalid for {term.name}"
                )
            if term.semantic_view == "front":
                region_name = (
                    "subject_left_nose_wing"
                    if term.boundary_name == "subject-left-alar"
                    else "subject_right_nose_wing"
                )
                expected_support = (
                    prepared.region_masks[region_name]
                    & prepared.support_mask
                    & ~prepared.protected_mask
                )
            else:
                expected_support = (
                    (
                        prepared.region_masks["nose_tip"]
                        | prepared.region_masks["tip_alar_transition"]
                        | prepared.region_masks["subject_left_nose_wing"]
                        | prepared.region_masks["subject_right_nose_wing"]
                    )
                    & prepared.support_mask
                    & ~prepared.protected_mask
                )
            if not np.array_equal(
                term.support_vertex_indices,
                np.flatnonzero(expected_support),
            ):
                raise ValueError(
                    f"objective semantic support does not match {term.name}"
                )
        depth_support = np.asarray(self.depth_support_vertex_indices)
        if (
            depth_support.ndim != 1
            or not len(depth_support)
            or not np.issubdtype(depth_support.dtype, np.integer)
            or np.any(depth_support < 0)
            or np.any(depth_support >= prepared.vertex_count)
            or np.any(np.diff(depth_support) <= 0)
            or np.any(prepared.protected_mask[depth_support])
            or np.any(~prepared.support_mask[depth_support])
        ):
            raise ValueError(
                "depth support indices must be sorted unique unprotected "
                "nasal support vertices"
            )
        if not np.array_equal(
            depth_support,
            np.flatnonzero(
                prepared.support_mask & ~prepared.protected_mask
            ),
        ):
            raise ValueError(
                "depth support indices must cover the fixed unprotected "
                "nasal support"
            )
        centers = np.asarray(self.smoothness_centers)
        neighbors = np.asarray(self.smoothness_neighbors)
        offsets = np.asarray(self.smoothness_neighbor_offsets)
        if (
            centers.ndim != 1
            or neighbors.ndim != 1
            or offsets.shape != (len(centers) + 1,)
            or not len(centers)
            or not len(neighbors)
            or not all(
                np.issubdtype(value.dtype, np.integer)
                for value in (centers, neighbors, offsets)
            )
        ):
            raise ValueError(
                "smoothness topology arrays must be non-empty integer "
                "vectors with valid offsets"
            )
        if (
            offsets[0] != 0
            or offsets[-1] != len(neighbors)
            or np.any(np.diff(offsets) <= 0)
            or np.any(centers < 0)
            or np.any(centers >= prepared.vertex_count)
            or np.any(neighbors < 0)
            or np.any(neighbors >= prepared.vertex_count)
            or np.any(np.diff(centers) <= 0)
        ):
            raise ValueError("smoothness topology indices or offsets are invalid")
        if (
            np.any(~prepared.support_mask[centers])
            or np.any(~prepared.support_mask[neighbors])
        ):
            raise ValueError(
                "smoothness topology must remain inside nasal support"
            )
        edge_count = int(self.smoothness_edge_count)
        if edge_count < 1 or len(neighbors) != 2 * edge_count:
            raise ValueError(
                "smoothness_edge_count must match fixed support edges"
            )
        (
            expected_orientation_indices,
            expected_orientation_faces,
            expected_orientation_cross,
            expected_orientation_inverse,
        ) = _orientation_reference_data(
            baseline,
            prepared,
            flame_basis,
        )
        orientation_indices = np.asarray(self.orientation_face_indices)
        orientation_faces = np.asarray(self.orientation_face_vertices)
        orientation_cross = np.asarray(
            self.orientation_reference_cross,
            dtype=np.float64,
        )
        orientation_inverse = np.asarray(
            self.orientation_reference_inverse_squared_norm,
            dtype=np.float64,
        )
        active_face_count = len(expected_orientation_indices)
        if (
            orientation_indices.ndim != 1
            or not np.issubdtype(orientation_indices.dtype, np.integer)
            or orientation_faces.shape != (active_face_count, 3)
            or not np.issubdtype(orientation_faces.dtype, np.integer)
            or orientation_cross.shape != (active_face_count, 3)
            or orientation_inverse.shape != (active_face_count,)
            or not np.isfinite(orientation_cross).all()
            or not np.isfinite(orientation_inverse).all()
            or np.any(orientation_inverse <= 0.0)
        ):
            raise ValueError(
                "orientation reference arrays have invalid shapes or values"
            )
        if (
            not np.array_equal(
                orientation_indices,
                expected_orientation_indices,
            )
            or not np.array_equal(
                orientation_faces,
                expected_orientation_faces,
            )
        ):
            raise ValueError(
                "orientation active faces do not match movable basis support"
            )
        if (
            not np.array_equal(
                orientation_cross,
                expected_orientation_cross,
            )
            or not np.array_equal(
                orientation_inverse,
                expected_orientation_inverse,
            )
        ):
            raise ValueError(
                "orientation reference data do not match baseline faces"
            )
        ordering = tuple(str(value) for value in self.parameter_ordering)
        expected_ordering = tuple(
            f"observable_flame_{index}" for index in range(rank)
        ) + tuple(NASAL_SEMANTIC_MODE_NAMES)
        if ordering != expected_ordering:
            raise ValueError(
                "parameter_ordering must list observable modes then "
                "canonical semantic modes"
            )
        lower_bounds = np.asarray(
            self.parameter_lower_bounds,
            dtype=np.float64,
        )
        upper_bounds = np.asarray(
            self.parameter_upper_bounds,
            dtype=np.float64,
        )
        expected_bounds = np.r_[
            np.full(rank, _FIXED_OBSERVABLE_COEFFICIENT_BOUND),
            np.full(
                len(NASAL_SEMANTIC_MODE_NAMES),
                _FIXED_SEMANTIC_COEFFICIENT_BOUND,
            ),
        ]
        if (
            lower_bounds.shape != (len(expected_ordering),)
            or upper_bounds.shape != (len(expected_ordering),)
            or not np.isfinite(lower_bounds).all()
            or not np.isfinite(upper_bounds).all()
            or not np.array_equal(lower_bounds, -expected_bounds)
            or not np.array_equal(upper_bounds, expected_bounds)
        ):
            raise ValueError(
                "parameter bounds must use the fixed subject-independent "
                "observable and semantic limits"
            )
        object.__setattr__(
            self,
            "baseline_vertices",
            _readonly_array(baseline, baseline.dtype),
        )
        object.__setattr__(
            self,
            "observable_vertex_basis",
            _readonly_array(flame_basis, np.float64),
        )
        object.__setattr__(self, "image_terms", terms)
        object.__setattr__(
            self,
            "depth_support_vertex_indices",
            _readonly_array(depth_support, np.int64),
        )
        object.__setattr__(
            self,
            "flame_mode_standard_deviations",
            _readonly_array(standard_deviations, np.float64),
        )
        object.__setattr__(
            self,
            "smoothness_centers",
            _readonly_array(centers, np.int64),
        )
        object.__setattr__(
            self,
            "smoothness_neighbors",
            _readonly_array(neighbors, np.int64),
        )
        object.__setattr__(
            self,
            "smoothness_neighbor_offsets",
            _readonly_array(offsets, np.int64),
        )
        object.__setattr__(self, "smoothness_edge_count", edge_count)
        object.__setattr__(
            self,
            "orientation_face_indices",
            _readonly_array(orientation_indices, np.int64),
        )
        object.__setattr__(
            self,
            "orientation_face_vertices",
            _readonly_array(orientation_faces, np.int64),
        )
        object.__setattr__(
            self,
            "orientation_reference_cross",
            _readonly_array(orientation_cross, np.float64),
        )
        object.__setattr__(
            self,
            "orientation_reference_inverse_squared_norm",
            _readonly_array(orientation_inverse, np.float64),
        )
        object.__setattr__(self, "parameter_ordering", ordering)
        object.__setattr__(
            self,
            "parameter_lower_bounds",
            _readonly_array(lower_bounds, np.float64),
        )
        object.__setattr__(
            self,
            "parameter_upper_bounds",
            _readonly_array(upper_bounds, np.float64),
        )

    @property
    def observable_rank(self) -> int:
        return self.observable_vertex_basis.shape[2]

    @property
    def parameter_count(self) -> int:
        return self.observable_rank + len(NASAL_SEMANTIC_MODE_NAMES)

    @property
    def orientation_active_face_count(self) -> int:
        return len(self.orientation_face_indices)

    @property
    def image_term_names(self) -> Tuple[str, ...]:
        return _OBJECTIVE_IMAGE_TERM_NAMES


def _freeze_objective_value(value):
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {
                str(key): _freeze_objective_value(item)
                for key, item in value.items()
            }
        )
    if isinstance(value, list):
        return tuple(_freeze_objective_value(item) for item in value)
    if isinstance(value, tuple):
        return tuple(_freeze_objective_value(item) for item in value)
    return value


@dataclass(frozen=True)
class SoftProfileNasalSlots:
    """Continuous semantic-profile slots for one objective image term."""

    name: str
    semantic_view: str
    boundary_name: str
    pixel_xy: np.ndarray
    depth: np.ndarray
    confidence: np.ndarray
    target_slot_y: np.ndarray
    support_vertex_indices: np.ndarray
    soft_weights: np.ndarray
    front_depth: np.ndarray
    soft_visibility: np.ndarray
    direction_sign: int

    def __post_init__(self) -> None:
        valid_specs = {
            name: (semantic_view, boundary_name)
            for name, semantic_view, boundary_name
            in _OBJECTIVE_IMAGE_TERM_SPECS
        }
        name = str(self.name)
        semantic_view = str(self.semantic_view)
        boundary_name = str(self.boundary_name)
        if (
            name not in valid_specs
            or valid_specs[name] != (semantic_view, boundary_name)
        ):
            raise ValueError("soft profile slot identity is invalid")
        pixels = np.asarray(self.pixel_xy, dtype=np.float64)
        depth = np.asarray(self.depth, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        slot_y = np.asarray(self.target_slot_y, dtype=np.float64)
        support = np.asarray(self.support_vertex_indices)
        weights = np.asarray(self.soft_weights, dtype=np.float64)
        front_depth = np.asarray(self.front_depth, dtype=np.float64)
        visibility = np.asarray(self.soft_visibility, dtype=np.float64)
        count = len(pixels)
        if (
            pixels.ndim != 2
            or pixels.shape[1:] != (2,)
            or depth.shape != (count,)
            or confidence.shape != (count,)
            or slot_y.shape != (count,)
            or weights.shape != (count, len(support))
            or front_depth.shape != (count,)
            or visibility.shape != weights.shape
            or not all(
                np.isfinite(value).all()
                for value in (
                    pixels,
                    depth,
                    confidence,
                    slot_y,
                    weights,
                    front_depth,
                    visibility,
                )
            )
            or np.any((confidence < 0.0) | (confidence > 1.0))
            or np.any(weights < 0.0)
            or np.any((visibility < 0.0) | (visibility > 1.0))
            or not np.allclose(
                np.sum(weights, axis=1),
                np.ones(count),
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise ValueError("soft profile slot arrays are invalid")
        if (
            support.ndim != 1
            or not len(support)
            or not np.issubdtype(support.dtype, np.integer)
            or np.any(support < 0)
            or np.any(np.diff(support) <= 0)
        ):
            raise ValueError("soft profile support indices are invalid")
        direction = int(self.direction_sign)
        if (
            isinstance(self.direction_sign, (bool, np.bool_))
            or direction not in (-1, 1)
        ):
            raise ValueError("soft profile direction_sign must be -1 or 1")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "semantic_view", semantic_view)
        object.__setattr__(self, "boundary_name", boundary_name)
        object.__setattr__(self, "pixel_xy", _readonly_array(pixels, np.float64))
        object.__setattr__(self, "depth", _readonly_array(depth, np.float64))
        object.__setattr__(
            self,
            "confidence",
            _readonly_array(confidence, np.float64),
        )
        object.__setattr__(
            self,
            "target_slot_y",
            _readonly_array(slot_y, np.float64),
        )
        object.__setattr__(
            self,
            "support_vertex_indices",
            _readonly_array(support, np.int64),
        )
        object.__setattr__(
            self,
            "soft_weights",
            _readonly_array(weights, np.float64),
        )
        object.__setattr__(
            self,
            "front_depth",
            _readonly_array(front_depth, np.float64),
        )
        object.__setattr__(
            self,
            "soft_visibility",
            _readonly_array(visibility, np.float64),
        )
        object.__setattr__(self, "direction_sign", direction)


@dataclass(frozen=True)
class MultiviewNasalSoftProjection:
    """Immutable C2 soft semantic-slot projection diagnostics."""

    candidate_vertices: np.ndarray
    per_term: Tuple[SoftProfileNasalSlots, ...]

    def __post_init__(self) -> None:
        vertices = _validate_vertices(self.candidate_vertices)
        terms = tuple(self.per_term)
        if (
            tuple(term.name for term in terms)
            != _OBJECTIVE_IMAGE_TERM_NAMES
            or not all(isinstance(term, SoftProfileNasalSlots) for term in terms)
        ):
            raise ValueError(
                "soft projection terms must use the fixed canonical order"
            )
        if any(
            np.any(term.support_vertex_indices >= len(vertices))
            for term in terms
        ):
            raise ValueError(
                "soft projection support indices exceed candidate vertices"
            )
        object.__setattr__(
            self,
            "candidate_vertices",
            _readonly_array(vertices, vertices.dtype),
        )
        object.__setattr__(self, "per_term", terms)

    @property
    def by_term(self) -> Mapping[str, SoftProfileNasalSlots]:
        return MappingProxyType({term.name: term for term in self.per_term})


@dataclass(frozen=True)
class MultiviewNasalResidualEvaluation:
    """Copy-safe minimal output for iterative least-squares solvers."""

    residuals: np.ndarray
    term_slices: Mapping[str, slice]
    term_residuals: Mapping[str, np.ndarray]

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residuals, dtype=np.float64)
        expected_names = (
            _OBJECTIVE_IMAGE_TERM_NAMES
            + _OBJECTIVE_REGULARIZATION_TERM_NAMES
        )
        slices = dict(self.term_slices)
        terms = dict(self.term_residuals)
        if (
            residuals.ndim != 1
            or not np.isfinite(residuals).all()
            or tuple(slices) != expected_names
            or tuple(terms) != expected_names
        ):
            raise ValueError("minimal objective residual output is invalid")
        cursor = 0
        frozen_terms = {}
        frozen_slices = {}
        for name in expected_names:
            term_slice = slices[name]
            term = np.asarray(terms[name], dtype=np.float64)
            if (
                not isinstance(term_slice, slice)
                or term_slice.step not in (None, 1)
                or term_slice.start != cursor
                or term_slice.stop is None
                or term.shape != (term_slice.stop - cursor,)
                or not np.array_equal(term, residuals[term_slice])
            ):
                raise ValueError("minimal objective term layout is invalid")
            frozen_terms[name] = _readonly_array(term, np.float64)
            frozen_slices[name] = slice(cursor, term_slice.stop)
            cursor = term_slice.stop
        if cursor != len(residuals):
            raise ValueError("minimal objective terms do not cover residuals")
        object.__setattr__(
            self,
            "residuals",
            _readonly_array(residuals, np.float64),
        )
        object.__setattr__(
            self,
            "term_slices",
            MappingProxyType(frozen_slices),
        )
        object.__setattr__(
            self,
            "term_residuals",
            MappingProxyType(frozen_terms),
        )

    @property
    def residual_vector(self) -> np.ndarray:
        return self.residuals


@dataclass(frozen=True)
class MultiviewNasalObjectiveResult:
    """One immutable unified residual evaluation and optimization report."""

    residuals: np.ndarray
    term_slices: Mapping[str, slice]
    term_residuals: Mapping[str, np.ndarray]
    raw_costs: Mapping[str, float]
    robust_costs: Mapping[str, float]
    total_raw_cost: float
    total_robust_cost: float
    candidate: CandidateNasalMesh
    projection: MultiviewNasalSoftProjection
    effective_observation_counts: Mapping[str, int]
    sample_counts: Mapping[str, int]
    effective_confidence_sums: Mapping[str, float]
    per_view_effective_observation_counts: Mapping[str, int]
    per_view_sample_counts: Mapping[str, int]
    per_view_effective_confidence_sums: Mapping[str, float]
    symmetry_evidence_factors: Mapping[str, float]
    parameter_ordering: Tuple[str, ...]
    robust_loss: str
    robust_f_scale: float
    report_data: Mapping[str, object]

    def __post_init__(self) -> None:
        residuals = np.asarray(self.residuals, dtype=np.float64)
        if residuals.ndim != 1 or not np.isfinite(residuals).all():
            raise ValueError("objective residuals must be a finite vector")
        expected_names = (
            _OBJECTIVE_IMAGE_TERM_NAMES
            + _OBJECTIVE_REGULARIZATION_TERM_NAMES
        )
        slices = dict(self.term_slices)
        residual_mapping = dict(self.term_residuals)
        if (
            tuple(slices) != expected_names
            or tuple(residual_mapping) != expected_names
        ):
            raise ValueError(
                "objective terms must use the fixed canonical order"
            )
        cursor = 0
        residual_snapshots = {}
        slice_snapshots = {}
        for name in expected_names:
            term_slice = slices[name]
            if (
                not isinstance(term_slice, slice)
                or term_slice.step not in (None, 1)
                or term_slice.start != cursor
                or term_slice.stop is None
                or term_slice.stop < cursor
            ):
                raise ValueError("objective term slices must be contiguous")
            term = np.asarray(residual_mapping[name], dtype=np.float64)
            if (
                term.shape != (term_slice.stop - cursor,)
                or not np.isfinite(term).all()
                or not np.array_equal(term, residuals[term_slice])
            ):
                raise ValueError(
                    f"objective term residuals are invalid for {name}"
                )
            snapshot = _readonly_array(term, np.float64)
            residual_snapshots[name] = snapshot
            slice_snapshots[name] = slice(cursor, term_slice.stop)
            cursor = term_slice.stop
        if cursor != len(residuals):
            raise ValueError("objective term slices must cover the residual vector")
        if not isinstance(self.candidate, CandidateNasalMesh):
            raise ValueError("candidate must be a CandidateNasalMesh")
        if not isinstance(self.projection, MultiviewNasalSoftProjection):
            raise ValueError(
                "projection must be a MultiviewNasalSoftProjection"
            )
        if not np.array_equal(
            self.candidate.vertices,
            self.projection.candidate_vertices,
        ):
            raise ValueError("candidate and projection vertices must match")
        raw_costs = {
            str(name): float(value)
            for name, value in self.raw_costs.items()
        }
        robust_costs = {
            str(name): float(value)
            for name, value in self.robust_costs.items()
        }
        if (
            tuple(raw_costs) != expected_names
            or tuple(robust_costs) != expected_names
            or not np.isfinite(
                np.asarray(
                    tuple(raw_costs.values())
                    + tuple(robust_costs.values()),
                    dtype=np.float64,
                )
            ).all()
            or any(value < 0.0 for value in raw_costs.values())
            or any(value < 0.0 for value in robust_costs.values())
        ):
            raise ValueError("objective term costs are invalid")
        total_raw = _finite_real("total_raw_cost", self.total_raw_cost)
        total_robust = _finite_real(
            "total_robust_cost",
            self.total_robust_cost,
        )
        if not np.isclose(
            total_raw,
            sum(raw_costs.values()),
            atol=1e-12,
            rtol=1e-12,
        ) or not np.isclose(
            total_robust,
            sum(robust_costs.values()),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("objective total costs do not match term costs")
        if str(self.robust_loss) != "soft_l1":
            raise ValueError("objective robust_loss must be soft_l1")
        robust_scale = _finite_real(
            "robust_f_scale",
            self.robust_f_scale,
            strictly_positive=True,
        )
        object.__setattr__(
            self,
            "residuals",
            _readonly_array(residuals, np.float64),
        )
        object.__setattr__(
            self,
            "term_slices",
            MappingProxyType(slice_snapshots),
        )
        object.__setattr__(
            self,
            "term_residuals",
            MappingProxyType(residual_snapshots),
        )
        object.__setattr__(
            self,
            "raw_costs",
            MappingProxyType(raw_costs),
        )
        object.__setattr__(
            self,
            "robust_costs",
            MappingProxyType(robust_costs),
        )
        object.__setattr__(self, "total_raw_cost", total_raw)
        object.__setattr__(self, "total_robust_cost", total_robust)
        count_mappings = {}
        for name in (
            "effective_observation_counts",
            "sample_counts",
        ):
            values = {
                str(key): int(value)
                for key, value in getattr(self, name).items()
            }
            if tuple(values) != _OBJECTIVE_IMAGE_TERM_NAMES or any(
                value < 0 for value in values.values()
            ):
                raise ValueError(f"{name} must cover all image terms")
            count_mappings[name] = values
            object.__setattr__(self, name, MappingProxyType(values))
        confidence_sums = {
            str(key): float(value)
            for key, value in self.effective_confidence_sums.items()
        }
        if (
            tuple(confidence_sums) != _OBJECTIVE_IMAGE_TERM_NAMES
            or not np.isfinite(
                np.asarray(tuple(confidence_sums.values()))
            ).all()
            or any(value < 0.0 for value in confidence_sums.values())
        ):
            raise ValueError(
                "effective_confidence_sums must cover all image terms"
            )
        terms_by_view = {
            semantic_view: tuple(
                name
                for name, view, _boundary in _OBJECTIVE_IMAGE_TERM_SPECS
                if view == semantic_view
            )
            for semantic_view in NASAL_VIEWS
        }
        expected_per_view_counts = {
            name: {
                view: sum(
                    count_mappings[name][term]
                    for term in terms_by_view[view]
                )
                for view in NASAL_VIEWS
            }
            for name in count_mappings
        }
        for name in (
            "per_view_effective_observation_counts",
            "per_view_sample_counts",
        ):
            values = {
                str(key): int(value)
                for key, value in getattr(self, name).items()
            }
            source_name = name.replace("per_view_", "")
            if (
                tuple(values) != tuple(NASAL_VIEWS)
                or values != expected_per_view_counts[source_name]
            ):
                raise ValueError(
                    f"{name} must aggregate the canonical image terms"
                )
            object.__setattr__(self, name, MappingProxyType(values))
        per_view_confidence_sums = {
            str(key): float(value)
            for key, value in (
                self.per_view_effective_confidence_sums.items()
            )
        }
        expected_view_confidence = {
            view: sum(
                confidence_sums[term]
                for term in terms_by_view[view]
            )
            for view in NASAL_VIEWS
        }
        if (
            tuple(per_view_confidence_sums) != tuple(NASAL_VIEWS)
            or not all(
                np.isclose(
                    per_view_confidence_sums[view],
                    expected_view_confidence[view],
                    atol=1e-12,
                    rtol=1e-12,
                )
                for view in NASAL_VIEWS
            )
        ):
            raise ValueError(
                "per_view_effective_confidence_sums must aggregate "
                "the canonical image terms"
            )
        factors = {
            str(key): float(value)
            for key, value in self.symmetry_evidence_factors.items()
        }
        if (
            tuple(factors) != _ASYMMETRY_MODE_NAMES
            or not np.isfinite(np.asarray(tuple(factors.values()))).all()
            or any(value <= 0.0 for value in factors.values())
        ):
            raise ValueError("symmetry evidence factors are invalid")
        object.__setattr__(
            self,
            "effective_confidence_sums",
            MappingProxyType(confidence_sums),
        )
        object.__setattr__(
            self,
            "per_view_effective_confidence_sums",
            MappingProxyType(per_view_confidence_sums),
        )
        object.__setattr__(
            self,
            "symmetry_evidence_factors",
            MappingProxyType(factors),
        )
        object.__setattr__(
            self,
            "parameter_ordering",
            tuple(str(value) for value in self.parameter_ordering),
        )
        object.__setattr__(self, "robust_loss", "soft_l1")
        object.__setattr__(self, "robust_f_scale", robust_scale)
        object.__setattr__(
            self,
            "report_data",
            _freeze_objective_value(self.report_data),
        )

    @property
    def residual_vector(self) -> np.ndarray:
        return self.residuals


def _validate_objective_observable(
    observable_flame,
    vertex_count: int,
) -> Tuple[np.ndarray, int]:
    vertex_basis = _validate_basis_array(
        "observable FLAME vertex_basis",
        getattr(observable_flame, "vertex_basis", None),
        (vertex_count, 3, None),
    )
    rank = vertex_basis.shape[2]
    if int(getattr(observable_flame, "retained_rank", rank)) != rank:
        raise ValueError(
            "observable FLAME retained rank does not match vertex_basis"
        )
    coefficient_basis = _validate_basis_array(
        "observable FLAME coefficient_basis",
        getattr(observable_flame, "coefficient_basis", None),
        (None, rank),
    )
    if rank and not np.allclose(
        coefficient_basis.T @ coefficient_basis,
        np.eye(rank),
        atol=1e-10,
        rtol=0.0,
    ):
        raise ValueError(
            "observable FLAME coefficient_basis columns must be orthonormal "
            "in original coefficient space"
        )
    return vertex_basis, rank


def _validate_prepared_observations(
    prepared: PreparedNasalProjectionContext,
    observations: NasalObservationBundle,
) -> None:
    if not isinstance(observations, NasalObservationBundle):
        raise ValueError("observations must be a NasalObservationBundle")
    for contract in prepared.view_contracts:
        observation = observations.by_view[contract.semantic_view]
        if (
            tuple(observation.work_size) != contract.work_size
            or tuple(float(value) for value in observation.roi_work_xyxy)
            != contract.roi_work_xyxy
            or tuple(sorted(str(name) for name in observation.distance_fields))
            != contract.target_names
            or not np.array_equal(
                observation.confidence,
                contract.confidence,
            )
        ):
            raise ValueError(
                f"{contract.semantic_view} observations do not match the "
                "prepared projection context"
            )


def _smoothness_topology(
    prepared: PreparedNasalProjectionContext,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    support_edges = prepared.edge_vertices[
        prepared.support_mask[prepared.edge_vertices[:, 0]]
        & prepared.support_mask[prepared.edge_vertices[:, 1]]
    ]
    if not len(support_edges):
        raise ValueError(
            "nasal support must contain at least one topology edge for "
            "surface smoothness"
        )
    adjacency = {}
    for first, second in support_edges:
        first_index = int(first)
        second_index = int(second)
        adjacency.setdefault(first_index, []).append(second_index)
        adjacency.setdefault(second_index, []).append(first_index)
    centers = np.asarray(sorted(adjacency), dtype=np.int64)
    neighbors = []
    offsets = [0]
    for center in centers:
        neighbors.extend(sorted(set(adjacency[int(center)])))
        offsets.append(len(neighbors))
    return (
        centers,
        np.asarray(neighbors, dtype=np.int64),
        np.asarray(offsets, dtype=np.int64),
        int(len(support_edges)),
    )


def _orientation_reference_data(
    baseline_vertices: np.ndarray,
    prepared: PreparedNasalProjectionContext,
    observable_vertex_basis: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return static baseline references for every potentially moved face."""
    baseline = np.asarray(baseline_vertices)
    observable = np.asarray(observable_vertex_basis, dtype=np.float64)
    movable_vertices = np.asarray(prepared.support_mask, dtype=bool).copy()
    if observable.shape[2]:
        movable_vertices |= np.any(observable != 0.0, axis=(1, 2))
    movable_vertices |= np.any(
        prepared.semantic_vectors != 0.0,
        axis=(0, 2),
    )
    candidate_indices = np.flatnonzero(
        np.any(movable_vertices[prepared.faces], axis=1)
    )
    if not len(candidate_indices):
        raise ValueError(
            "orientation barrier requires at least one potentially "
            "deformable baseline face"
        )
    candidate_faces = prepared.faces[candidate_indices]
    triangles = np.asarray(
        baseline[candidate_faces],
        dtype=np.float64,
    )
    reference_cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    squared_norm = np.einsum(
        "fi,fi->f",
        reference_cross,
        reference_cross,
        optimize=True,
    )
    edge_vectors = np.stack(
        (
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 1],
            triangles[:, 0] - triangles[:, 2],
        ),
        axis=1,
    )
    max_edge_squared = np.max(
        np.einsum(
            "fei,fei->fe",
            edge_vectors,
            edge_vectors,
            optimize=True,
        ),
        axis=1,
    )
    local_quality = np.divide(
        np.sqrt(squared_norm),
        max_edge_squared,
        out=np.full_like(squared_norm, np.nan),
        where=max_edge_squared > 0.0,
    )
    degenerate = (
        ~np.isfinite(local_quality)
        | (
            local_quality
            <= _ORIENTATION_BASELINE_LOCAL_QUALITY_THRESHOLD
        )
    )
    if np.any(degenerate):
        degenerate_indices = candidate_indices[degenerate]
        preview = ", ".join(
            str(int(index))
            for index in degenerate_indices[
                :_ORIENTATION_DEGENERATE_FACE_PREVIEW_COUNT
            ]
        )
        suffix = (
            ", ..."
            if len(degenerate_indices)
            > _ORIENTATION_DEGENERATE_FACE_PREVIEW_COUNT
            else ""
        )
        raise ValueError(
            "orientation barrier found degenerate potentially "
            "deformable baseline faces: "
            f"total={len(degenerate_indices)}, "
            f"indices=[{preview}{suffix}], "
            "local_quality_threshold="
            f"{_ORIENTATION_BASELINE_LOCAL_QUALITY_THRESHOLD:g}"
        )
    active_indices = candidate_indices.astype(
        np.int64,
        copy=False,
    )
    active_faces = candidate_faces.astype(
        np.int64,
        copy=False,
    )
    active_cross = reference_cross
    inverse_squared_norm = 1.0 / squared_norm
    return (
        active_indices,
        active_faces,
        active_cross,
        inverse_squared_norm,
    )


def _canonical_curve_quantile_points(
    curve: np.ndarray,
    quantiles: np.ndarray,
) -> np.ndarray:
    """Sample an observation curve at deterministic oriented arclength slots."""
    values = np.asarray(curve, dtype=np.float64)
    if (
        values.ndim != 2
        or values.shape[1:] != (2,)
        or not len(values)
        or not np.isfinite(values).all()
    ):
        raise ValueError(
            "observation target curves must have finite shape (N, 2)"
        )
    if len(values) == 1:
        return np.repeat(values, len(quantiles), axis=0)
    first_key = (float(values[0, 1]), float(values[0, 0]))
    last_key = (float(values[-1, 1]), float(values[-1, 0]))
    if first_key > last_key:
        values = values[::-1]
    segment_lengths = np.linalg.norm(np.diff(values, axis=0), axis=1)
    usable = segment_lengths > 1e-12
    if not np.any(usable):
        return np.repeat(values[:1], len(quantiles), axis=0)
    starts = values[:-1][usable]
    ends = values[1:][usable]
    lengths = segment_lengths[usable]
    cumulative = np.cumsum(lengths)
    targets = np.asarray(quantiles, dtype=np.float64) * cumulative[-1]
    segment_indices = np.searchsorted(cumulative, targets, side="right")
    segment_indices = np.minimum(segment_indices, len(lengths) - 1)
    previous = np.concatenate(([0.0], cumulative[:-1]))
    alpha = (
        targets - previous[segment_indices]
    ) / lengths[segment_indices]
    return (
        (1.0 - alpha[:, None]) * starts[segment_indices]
        + alpha[:, None] * ends[segment_indices]
    )


def prepare_multiview_nasal_objective_context(
    baseline_vertices: np.ndarray,
    observable_flame,
    semantic_basis,
    observations: NasalObservationBundle,
    *,
    faces: Optional[np.ndarray] = None,
    views: Optional[Sequence[ProjectionView]] = None,
    projection_context: Optional[PreparedNasalProjectionContext] = None,
    sampling_config: Optional[MultiviewNasalSamplingConfig] = None,
    flame_mode_standard_deviations: Optional[Sequence[float]] = None,
) -> MultiviewNasalObjectiveContext:
    """Prepare fixed image slots, priors, and support topology once."""
    baseline = _validate_vertices(baseline_vertices)
    if projection_context is None:
        if faces is None or views is None:
            raise ValueError(
                "faces and views are required without projection_context"
            )
        prepared = prepare_nasal_projection_context(
            faces,
            semantic_basis,
            observations,
            views,
            config=sampling_config,
        )
    else:
        if not isinstance(
            projection_context,
            PreparedNasalProjectionContext,
        ):
            raise ValueError(
                "projection_context must be a PreparedNasalProjectionContext"
            )
        if sampling_config is not None:
            raise ValueError(
                "sampling_config must be omitted with projection_context"
            )
        prepared = projection_context
        if faces is not None and not np.array_equal(faces, prepared.faces):
            raise ValueError("faces do not match projection_context")
        if views is not None:
            canonical_views = _canonical_views(views)
            for supplied, existing in zip(
                canonical_views,
                prepared.views,
            ):
                if not all(
                    np.array_equal(
                        getattr(supplied, name),
                        getattr(existing, name),
                    )
                    for name in (
                        "K",
                        "R_model_to_camera",
                        "t_model_to_camera",
                    )
                ):
                    raise ValueError("views do not match projection_context")
    if len(baseline) != prepared.vertex_count:
        raise ValueError(
            "baseline vertex count does not match prepared projection context"
        )
    _validate_prepared_observations(prepared, observations)
    semantic_vectors = _semantic_vectors(
        semantic_basis,
        prepared.vertex_count,
    )
    if not np.array_equal(semantic_vectors, prepared.semantic_vectors):
        raise ValueError(
            "semantic basis vectors do not match projection_context"
        )
    supplied_regions = _region_masks(
        semantic_basis,
        prepared.vertex_count,
        validate_semantic=False,
    )
    if (
        not np.array_equal(
            np.asarray(semantic_basis.support_mask, dtype=bool),
            prepared.support_mask,
        )
        or not np.array_equal(
            np.asarray(semantic_basis.protected_mask, dtype=bool),
            prepared.protected_mask,
        )
        or any(
            not np.array_equal(
                supplied_regions[name],
                prepared.region_masks[name],
            )
            for name in _REGION_NAMES
        )
    ):
        raise ValueError(
            "semantic basis masks do not match projection_context"
        )
    observable_basis, rank = _validate_objective_observable(
        observable_flame,
        prepared.vertex_count,
    )
    if flame_mode_standard_deviations is None:
        flame_standard_deviations = np.ones(rank, dtype=np.float64)
    else:
        flame_standard_deviations = np.asarray(
            _positive_tuple(
                "flame_mode_standard_deviations",
                flame_mode_standard_deviations,
                rank,
            ),
            dtype=np.float64,
        )
    unprotected = prepared.support_mask & ~prepared.protected_mask
    side_support = unprotected & (
        prepared.region_masks["nose_tip"]
        | prepared.region_masks["tip_alar_transition"]
        | prepared.region_masks["subject_left_nose_wing"]
        | prepared.region_masks["subject_right_nose_wing"]
    )
    support_by_term = {
        "front_subject_left_alar": (
            unprotected
            & prepared.region_masks["subject_left_nose_wing"]
        ),
        "front_subject_right_alar": (
            unprotected
            & prepared.region_masks["subject_right_nose_wing"]
        ),
        "subject_left_nasal_profile": side_support,
        "subject_right_nasal_profile": side_support,
    }
    if any(not np.any(mask) for mask in support_by_term.values()):
        raise ValueError(
            "every objective image term needs non-empty semantic support"
        )
    depth_support_indices = np.flatnonzero(unprotected).astype(np.int64)
    baseline_pixels_by_view = {}
    for semantic_view, view in zip(NASAL_VIEWS, prepared.views):
        projected = project_points_strict(
            baseline,
            view.K,
            view.R_model_to_camera,
            view.t_model_to_camera,
            epsilon=float(prepared.config.min_depth),
        )
        if np.any(
            projected.depth[depth_support_indices]
            <= float(prepared.config.min_depth)
        ):
            raise ValueError(
                f"baseline nasal support has invalid depth in {semantic_view}"
            )
        baseline_pixels_by_view[semantic_view] = np.asarray(
            projected.pixel_xy,
            dtype=np.float64,
        )
    image_terms = []
    for name, semantic_view, boundary_name in _OBJECTIVE_IMAGE_TERM_SPECS:
        observation = observations.by_view[semantic_view]
        count = (
            int(prepared.config.front_samples_per_region)
            if semantic_view == "front"
            else int(prepared.config.side_samples_per_view)
        )
        quantiles = (
            np.arange(count, dtype=np.float64) + 0.5
        ) / float(count)
        evidence_points = _canonical_curve_quantile_points(
            observation.boundaries_work[boundary_name],
            quantiles,
        )
        confidence, inside, _outside_distance = (
            _bilinear_sample_clamped(
                observation.confidence,
                evidence_points,
            )
        )
        if not np.all(inside):
            raise ValueError(
                f"{semantic_view} observation target {boundary_name} "
                "extends outside its image evidence domain"
            )
        target_polyline = np.asarray(
            observation.boundaries_work[boundary_name],
            dtype=np.float64,
        )
        support_indices = np.flatnonzero(
            support_by_term[name]
        ).astype(np.int64)
        baseline_pixels = baseline_pixels_by_view[semantic_view]
        baseline_center_x = float(
            np.median(baseline_pixels[depth_support_indices, 0])
        )
        target_offset = float(
            np.median(evidence_points[:, 0]) - baseline_center_x
        )
        if abs(target_offset) <= 1e-12:
            offsets = target_polyline[:, 0] - baseline_center_x
            target_offset = float(
                offsets[np.argmax(np.abs(offsets))]
            )
        direction_sign = 1 if target_offset >= 0.0 else -1
        image_terms.append(
            _PreparedObjectiveImageTerm(
                name=name,
                semantic_view=semantic_view,
                boundary_name=boundary_name,
                slot_quantiles=quantiles,
                target_slot_xy=evidence_points,
                target_polyline=target_polyline,
                support_vertex_indices=support_indices,
                direction_sign=direction_sign,
                distance_field=observation.distance_fields[boundary_name],
                confidence=confidence,
                work_size=tuple(observation.work_size),
                roi_work_xyxy=tuple(observation.roi_work_xyxy),
            )
        )
    centers, neighbors, offsets, edge_count = _smoothness_topology(prepared)
    (
        orientation_indices,
        orientation_faces,
        orientation_cross,
        orientation_inverse,
    ) = _orientation_reference_data(
        baseline,
        prepared,
        observable_basis,
    )
    ordering = tuple(
        f"observable_flame_{index}" for index in range(rank)
    ) + tuple(NASAL_SEMANTIC_MODE_NAMES)
    upper_bounds = np.r_[
        np.full(rank, _FIXED_OBSERVABLE_COEFFICIENT_BOUND),
        np.full(
            len(NASAL_SEMANTIC_MODE_NAMES),
            _FIXED_SEMANTIC_COEFFICIENT_BOUND,
        ),
    ]
    return MultiviewNasalObjectiveContext(
        baseline_vertices=baseline,
        observable_vertex_basis=observable_basis,
        projection_context=prepared,
        image_terms=tuple(image_terms),
        depth_support_vertex_indices=depth_support_indices,
        flame_mode_standard_deviations=flame_standard_deviations,
        smoothness_centers=centers,
        smoothness_neighbors=neighbors,
        smoothness_neighbor_offsets=offsets,
        smoothness_edge_count=edge_count,
        orientation_face_indices=orientation_indices,
        orientation_face_vertices=orientation_faces,
        orientation_reference_cross=orientation_cross,
        orientation_reference_inverse_squared_norm=orientation_inverse,
        parameter_ordering=ordering,
        parameter_lower_bounds=-upper_bounds,
        parameter_upper_bounds=upper_bounds,
    )


def _bilinear_sample_clamped(
    field: np.ndarray,
    pixels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = field.shape
    x = np.asarray(pixels[:, 0], dtype=np.float64)
    y = np.asarray(pixels[:, 1], dtype=np.float64)
    inside = (
        np.isfinite(x)
        & np.isfinite(y)
        & (x >= 0.0)
        & (x <= float(width - 1))
        & (y >= 0.0)
        & (y <= float(height - 1))
    )
    clipped_x = np.clip(x, 0.0, float(width - 1))
    clipped_y = np.clip(y, 0.0, float(height - 1))
    x0 = np.floor(clipped_x).astype(np.int64)
    y0 = np.floor(clipped_y).astype(np.int64)
    x1 = np.minimum(x0 + 1, width - 1)
    y1 = np.minimum(y0 + 1, height - 1)
    wx = clipped_x - x0
    wy = clipped_y - y0
    source = np.asarray(field, dtype=np.float64)
    values = (
        source[y0, x0] * (1.0 - wx) * (1.0 - wy)
        + source[y0, x1] * wx * (1.0 - wy)
        + source[y1, x0] * (1.0 - wx) * wy
        + source[y1, x1] * wx * wy
    )
    outside_distance = np.sqrt(
        (x - clipped_x) * (x - clipped_x)
        + (y - clipped_y) * (y - clipped_y)
    )
    return values, inside, outside_distance


def _stable_softplus(values: np.ndarray) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    return np.maximum(source, 0.0) + np.log1p(np.exp(-np.abs(source)))


def _stable_logsumexp(values: np.ndarray, axis: int) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    maximum = np.max(source, axis=axis, keepdims=True)
    result = maximum + np.log(
        np.sum(np.exp(source - maximum), axis=axis, keepdims=True)
    )
    return np.squeeze(result, axis=axis)


def _smooth_project_support(
    points: np.ndarray,
    view: ProjectionView,
    min_depth: float,
    depth_scale: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project finite points through a smooth positive-depth surrogate."""
    source = np.asarray(points, dtype=np.float64)
    with np.errstate(over="ignore", invalid="ignore"):
        camera = source @ view.R_model_to_camera.T + view.t_model_to_camera
    if not np.isfinite(camera).all():
        raise ValueError("camera projection produced non-finite camera points")
    raw_depth = camera[:, 2]
    scaled_depth = (raw_depth - float(min_depth)) / float(depth_scale)
    positive_depth = float(min_depth) + float(depth_scale) * _stable_softplus(
        scaled_depth
    )
    guarded_camera = np.array(camera, copy=True)
    guarded_camera[:, 2] = positive_depth
    with np.errstate(over="ignore", invalid="ignore"):
        homogeneous = guarded_camera @ view.K.T
    denominator = homogeneous[:, 2]
    if (
        not np.isfinite(homogeneous).all()
        or np.any(np.abs(denominator) <= np.finfo(np.float64).tiny)
    ):
        raise ValueError("guarded camera projection is numerically invalid")
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        pixels = homogeneous[:, :2] / denominator[:, None]
    if not np.isfinite(pixels).all():
        raise ValueError("guarded camera projection produced non-finite pixels")
    return pixels, raw_depth, positive_depth


def _soft_profile_slots(
    projected_support: np.ndarray,
    raw_depth: np.ndarray,
    slot_y: np.ndarray,
    direction_sign: int,
    config: MultiviewNasalObjectiveConfig,
    min_depth: float,
    *,
    work_size: Tuple[int, int],
    include_diagnostics: bool = True,
) -> Tuple[
    np.ndarray,
    Optional[np.ndarray],
    Optional[np.ndarray],
    np.ndarray,
    Optional[np.ndarray],
]:
    """Select continuous exterior slots with a local soft z-buffer."""
    pixels = np.asarray(projected_support, dtype=np.float64)
    depths = np.asarray(raw_depth, dtype=np.float64)
    target_y = np.asarray(slot_y, dtype=np.float64)
    if (
        len(work_size) != 2
        or any(
            isinstance(value, (bool, np.bool_))
            or not isinstance(value, (int, np.integer))
            or int(value) < 1
            for value in work_size
        )
    ):
        raise ValueError("work_size must contain two positive integers")
    work_width = float(work_size[0])
    horizontal_center = 0.5 * (work_width - 1.0)
    horizontal_scale = max(horizontal_center, 0.5)
    vertical = (
        pixels[None, :, 1] - target_y[:, None]
    ) / float(config.profile_vertical_sigma_px)
    vertical_logits = -0.5 * vertical * vertical
    bounded_horizontal = (
        horizontal_center
        + horizontal_scale
        * np.tanh(
            (pixels[:, 0] - horizontal_center) / horizontal_scale
        )
    )
    directional_logits = (
        float(direction_sign)
        * bounded_horizontal[None, :]
        / float(config.profile_softmax_temperature_px)
    )
    validity_logits = -_stable_softplus(
        (float(min_depth) - depths)
        / float(config.profile_depth_validity_scale)
    )
    foreground_logits = vertical_logits + validity_logits[None, :]
    foreground_normalizer = _stable_logsumexp(
        foreground_logits,
        axis=1,
    )
    front_depth = -float(config.profile_front_depth_temperature) * (
        _stable_logsumexp(
            foreground_logits
            - depths[None, :]
            / float(config.profile_front_depth_temperature),
            axis=1,
        )
        - foreground_normalizer
    )
    visibility_logits = (
        validity_logits[None, :]
        - _stable_softplus(
            (depths[None, :] - front_depth[:, None])
            / float(config.profile_depth_visibility_scale)
        )
    )
    anchor_logits = vertical_logits + visibility_logits
    preliminary_max = np.max(
        anchor_logits,
        axis=1,
        keepdims=True,
    )
    preliminary_exp = np.exp(anchor_logits - preliminary_max)
    preliminary_weights = preliminary_exp / np.sum(
        preliminary_exp,
        axis=1,
        keepdims=True,
    )
    preliminary_points = preliminary_weights @ pixels
    squared_spatial_distance = (
        np.sum(preliminary_points * preliminary_points, axis=1)[:, None]
        + np.sum(pixels * pixels, axis=1)[None, :]
        - 2.0 * (preliminary_points @ pixels.T)
    )
    squared_spatial_distance = np.maximum(
        squared_spatial_distance,
        0.0,
    )
    local_logits = (
        -0.5
        * squared_spatial_distance
        / float(config.profile_visibility_spatial_sigma_px) ** 2
    )
    final_logits = anchor_logits + local_logits + directional_logits
    row_max = np.max(final_logits, axis=1, keepdims=True)
    stabilized = np.exp(final_logits - row_max)
    sums = np.sum(stabilized, axis=1, keepdims=True)
    if (
        not np.isfinite(front_depth).all()
        or not np.isfinite(stabilized).all()
        or not np.isfinite(sums).all()
        or np.any(sums <= 0.0)
    ):
        raise ValueError("soft profile weights are numerically invalid")
    weights = stabilized / sums
    slot_pixels = weights @ pixels
    slot_depth = weights @ depths if include_diagnostics else None
    diagnostic_weights = weights if include_diagnostics else None
    soft_visibility = (
        np.exp(visibility_logits) if include_diagnostics else None
    )
    return (
        slot_pixels,
        slot_depth,
        diagnostic_weights,
        front_depth,
        soft_visibility,
    )


def _point_to_polyline_distance(
    points: np.ndarray,
    term: _PreparedObjectiveImageTerm,
) -> np.ndarray:
    """Exact unsigned Euclidean distance to immutable target segments."""
    query = np.asarray(points, dtype=np.float64)
    relative = query[:, None, :] - term.target_segment_starts[None, :, :]
    lengths = term.target_segment_length_squared
    denominator = np.where(lengths > 0.0, lengths, 1.0)
    fraction = np.sum(
        relative * term.target_segment_deltas[None, :, :],
        axis=2,
    ) / denominator[None, :]
    fraction = np.clip(fraction, 0.0, 1.0)
    closest = (
        term.target_segment_starts[None, :, :]
        + fraction[:, :, None] * term.target_segment_deltas[None, :, :]
    )
    squared = np.sum((query[:, None, :] - closest) ** 2, axis=2)
    return np.sqrt(np.min(squared, axis=1))


def _normalized_image_residual(
    distances: np.ndarray,
    confidence: np.ndarray,
    view_confidence_sum: float,
    term_weight: float,
) -> np.ndarray:
    if float(view_confidence_sum) == 0.0:
        return np.zeros(len(confidence), dtype=np.float64)
    return (
        float(term_weight)
        * np.sqrt(np.asarray(confidence, dtype=np.float64))
        * np.asarray(distances, dtype=np.float64)
        / np.sqrt(float(view_confidence_sum))
    )


def _soft_minimum_depth(
    depths: np.ndarray,
    temperature: float,
) -> float:
    values = np.asarray(depths, dtype=np.float64)
    return float(
        -float(temperature)
        * (
            _stable_logsumexp(
                -values / float(temperature),
                axis=0,
            )
            - np.log(float(len(values)))
        )
    )


def _orientation_signed_area_ratios(
    candidate_vertices: np.ndarray,
    context: MultiviewNasalObjectiveContext,
) -> np.ndarray:
    triangles = np.asarray(
        np.asarray(candidate_vertices)[
            context.orientation_face_vertices
        ],
        dtype=np.float64,
    )
    candidate_cross = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    ratios = (
        np.einsum(
            "fi,fi->f",
            candidate_cross,
            context.orientation_reference_cross,
            optimize=True,
        )
        * context.orientation_reference_inverse_squared_norm
    )
    if (
        ratios.shape != (context.orientation_active_face_count,)
        or not np.isfinite(ratios).all()
    ):
        raise ValueError(
            "surface orientation signed area ratios are non-finite"
        )
    return ratios


def _soft_minimum_orientation_ratio(
    signed_area_ratios: np.ndarray,
    temperature: float,
) -> float:
    values = np.asarray(signed_area_ratios, dtype=np.float64)
    scale = float(temperature)
    if (
        values.ndim != 1
        or not len(values)
        or not np.isfinite(values).all()
        or not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise ValueError(
            "orientation soft minimum requires finite nonempty ratios "
            "and positive temperature"
        )
    return float(
        -scale
        * (
            _stable_logsumexp(
                -values / scale,
                axis=0,
            )
            - np.log(float(len(values)))
        )
    )


def _surface_orientation_barrier_residuals(
    signed_area_ratios: np.ndarray,
    config: MultiviewNasalObjectiveConfig,
) -> Tuple[np.ndarray, float]:
    """Build per-face residuals plus exact-min global feasibility."""
    values = np.asarray(signed_area_ratios, dtype=np.float64)
    actual_minimum = float(np.min(values))
    per_face = (
        float(config.orientation_barrier_weight)
        * _stable_softplus(
            (
                float(config.orientation_barrier_margin)
                - values
            )
            / float(config.orientation_barrier_scale)
        )
        / np.sqrt(float(len(values)))
    )
    global_feasibility = (
        float(config.orientation_barrier_weight)
        * _stable_softplus(
            np.asarray(
                [
                    (
                        float(config.orientation_barrier_margin)
                        - actual_minimum
                    )
                    / float(config.orientation_barrier_scale)
                ],
                dtype=np.float64,
            )
        )
    )
    return (
        np.concatenate((per_face, global_feasibility)),
        actual_minimum,
    )


@dataclass(frozen=True)
class _SoftTermEvaluation:
    pixel_xy: np.ndarray
    depth: np.ndarray
    soft_weights: np.ndarray
    front_depth: np.ndarray
    soft_visibility: np.ndarray


@dataclass(frozen=True)
class _ObjectiveCoreEvaluation:
    candidate_vertices: np.ndarray
    residuals: np.ndarray
    term_slices: Mapping[str, slice]
    term_residuals: Mapping[str, np.ndarray]
    soft_terms: Mapping[str, _SoftTermEvaluation]
    effective_counts: Mapping[str, int]
    sample_counts: Mapping[str, int]
    confidence_sums: Mapping[str, float]
    per_view_effective_counts: Mapping[str, int]
    per_view_sample_counts: Mapping[str, int]
    per_view_confidence_sums: Mapping[str, float]
    symmetry_factors: Mapping[str, float]
    soft_minimum_depths: np.ndarray
    orientation_signed_area_ratios: np.ndarray
    orientation_actual_minimum_signed_area_ratio: float


def _soft_l1_cost(residuals: np.ndarray, f_scale: float) -> float:
    scaled = np.asarray(residuals, dtype=np.float64) / float(f_scale)
    return float(
        float(f_scale) ** 2
        * np.sum(np.sqrt(1.0 + scaled * scaled) - 1.0)
    )


def _symmetry_factor(
    first_reliability: float,
    second_reliability: float,
    config: MultiviewNasalObjectiveConfig,
) -> float:
    bilateral_evidence = np.sqrt(
        float(first_reliability) * float(second_reliability)
    )
    floor = float(config.symmetry_evidence_floor)
    ceiling = float(config.symmetry_evidence_ceiling)
    return float(
        floor + (ceiling - floor) * (1.0 - bilateral_evidence)
    )


def _objective_config(
    config: Optional[MultiviewNasalObjectiveConfig],
) -> MultiviewNasalObjectiveConfig:
    limits = MultiviewNasalObjectiveConfig() if config is None else config
    if not isinstance(limits, MultiviewNasalObjectiveConfig):
        raise ValueError("config must be a MultiviewNasalObjectiveConfig")
    return limits


def _evaluate_multiview_nasal_objective_core(
    theta: np.ndarray,
    context: MultiviewNasalObjectiveContext,
    config: Optional[MultiviewNasalObjectiveConfig],
    *,
    include_diagnostics: bool,
) -> _ObjectiveCoreEvaluation:
    if not isinstance(context, MultiviewNasalObjectiveContext):
        raise ValueError(
            "context must be a MultiviewNasalObjectiveContext"
        )
    limits = _objective_config(config)
    values = _validate_coefficients(
        "theta",
        theta,
        context.parameter_count,
    )
    rank = context.observable_rank
    flame_coefficients = values[:rank]
    semantic_coefficients = values[rank:]
    candidate_vertices = np.array(
        context.baseline_vertices,
        copy=True,
    )
    if np.any(values):
        candidate_vertices += np.einsum(
            "vcr,r->vc",
            context.observable_vertex_basis,
            flame_coefficients,
            optimize=True,
        ).astype(candidate_vertices.dtype, copy=False)
        candidate_vertices += np.einsum(
            "mvc,m->vc",
            context.projection_context.semantic_vectors,
            semantic_coefficients,
            optimize=True,
        ).astype(candidate_vertices.dtype, copy=False)
    if not np.isfinite(candidate_vertices).all():
        raise ValueError("candidate vertices are non-finite")
    orientation_ratios = _orientation_signed_area_ratios(
        candidate_vertices,
        context,
    )

    prepared = context.projection_context
    support_indices = context.depth_support_vertex_indices
    support_vertices = candidate_vertices[support_indices]
    projected_by_view = {}
    raw_depth_by_view = {}
    for semantic_view, view in zip(NASAL_VIEWS, prepared.views):
        pixels, raw_depth, _positive_depth = _smooth_project_support(
            support_vertices,
            view,
            float(prepared.config.min_depth),
            float(limits.depth_softplus_scale),
        )
        projected_by_view[semantic_view] = pixels
        raw_depth_by_view[semantic_view] = raw_depth

    confidence_sums = {
        term.name: float(np.sum(term.confidence))
        for term in context.image_terms
    }
    effective_counts = {
        term.name: int(np.count_nonzero(term.confidence > 0.0))
        for term in context.image_terms
    }
    sample_counts = {
        term.name: term.sample_count
        for term in context.image_terms
    }
    terms_by_view = {
        semantic_view: tuple(
            term
            for term in context.image_terms
            if term.semantic_view == semantic_view
        )
        for semantic_view in NASAL_VIEWS
    }
    per_view_confidence_sums = {
        semantic_view: float(
            sum(
                confidence_sums[term.name]
                for term in terms_by_view[semantic_view]
            )
        )
        for semantic_view in NASAL_VIEWS
    }
    per_view_effective_counts = {
        semantic_view: int(
            sum(
                effective_counts[term.name]
                for term in terms_by_view[semantic_view]
            )
        )
        for semantic_view in NASAL_VIEWS
    }
    per_view_sample_counts = {
        semantic_view: int(
            sum(
                sample_counts[term.name]
                for term in terms_by_view[semantic_view]
            )
        )
        for semantic_view in NASAL_VIEWS
    }
    image_residuals = {}
    soft_terms = {}
    reliability = {}
    for term in context.image_terms:
        local_indices = np.searchsorted(
            support_indices,
            term.support_vertex_indices,
        )
        support_pixels = projected_by_view[term.semantic_view][local_indices]
        support_raw_depth = raw_depth_by_view[
            term.semantic_view
        ][local_indices]
        (
            slot_pixels,
            slot_depth,
            weights,
            front_depth,
            soft_visibility,
        ) = _soft_profile_slots(
            support_pixels,
            support_raw_depth,
            term.slot_y,
            term.direction_sign,
            limits,
            float(prepared.config.min_depth),
            work_size=term.work_size,
            include_diagnostics=include_diagnostics,
        )
        distances = _point_to_polyline_distance(slot_pixels, term)
        term_weight = (
            float(limits.front_image_weight)
            if term.semantic_view == "front"
            else float(limits.side_image_weight)
        )
        view_confidence_sum = per_view_confidence_sums[
            term.semantic_view
        ]
        residual = _normalized_image_residual(
            distances,
            term.confidence,
            view_confidence_sum,
            term_weight,
        )
        image_residuals[term.name] = residual
        if include_diagnostics:
            soft_terms[term.name] = _SoftTermEvaluation(
                pixel_xy=slot_pixels,
                depth=slot_depth,
                soft_weights=weights,
                front_depth=front_depth,
                soft_visibility=soft_visibility,
            )
        reliability[term.name] = (
            confidence_sums[term.name] / float(term.sample_count)
        )

    raw_depths = np.concatenate(
        [raw_depth_by_view[view] for view in NASAL_VIEWS]
    )
    normalized_vertex_barrier = (
        float(limits.depth_barrier_weight)
        * _stable_softplus(
            (
                float(prepared.config.min_depth) - raw_depths
            )
            / float(limits.depth_barrier_scale)
        )
        / np.sqrt(float(len(raw_depths)))
    )
    soft_minimum_depths = np.asarray(
        [
            _soft_minimum_depth(
                raw_depth_by_view[view],
                float(limits.depth_barrier_softmin_temperature),
            )
            for view in NASAL_VIEWS
        ],
        dtype=np.float64,
    )
    global_depth_barrier = (
        float(limits.depth_barrier_weight)
        * _stable_softplus(
            (
                float(prepared.config.min_depth) - soft_minimum_depths
            )
            / float(limits.depth_barrier_scale)
        )
    )
    depth_barrier = np.concatenate(
        (normalized_vertex_barrier, global_depth_barrier)
    )

    flame_prior = (
        float(limits.flame_prior_weight)
        * flame_coefficients
        / context.flame_mode_standard_deviations
    )
    semantic_prior = (
        float(limits.semantic_prior_weight)
        * semantic_coefficients
        / np.asarray(
            limits.semantic_prior_standard_deviations,
            dtype=np.float64,
        )
    )
    displacement = candidate_vertices - context.baseline_vertices
    neighbor_sums = np.add.reduceat(
        displacement[context.smoothness_neighbors],
        context.smoothness_neighbor_offsets[:-1],
        axis=0,
    )
    degrees = np.diff(context.smoothness_neighbor_offsets).astype(
        np.float64
    )
    laplacian = (
        displacement[context.smoothness_centers]
        - neighbor_sums / degrees[:, None]
    )
    smoothness = (
        float(limits.smoothness_weight)
        * laplacian.reshape(-1)
        / (
            float(limits.smoothness_scale)
            * np.sqrt(float(context.smoothness_edge_count))
        )
    )
    (
        orientation_barrier,
        orientation_actual_minimum,
    ) = _surface_orientation_barrier_residuals(
        orientation_ratios,
        limits,
    )
    width_factor = _symmetry_factor(
        reliability["front_subject_left_alar"],
        reliability["front_subject_right_alar"],
        limits,
    )
    depth_factor = _symmetry_factor(
        reliability["subject_left_nasal_profile"],
        reliability["subject_right_nasal_profile"],
        limits,
    )
    symmetry_factors = {
        "alar_width_asymmetry": width_factor,
        "alar_depth_asymmetry": depth_factor,
    }
    symmetry = (
        float(limits.symmetry_weight)
        * np.asarray((width_factor, depth_factor), dtype=np.float64)
        * semantic_coefficients[[1, 3]]
    )
    named_residuals = dict(image_residuals)
    named_residuals.update(
        {
            "projection_depth_barrier": depth_barrier,
            "flame_prior": flame_prior,
            "semantic_prior": semantic_prior,
            "surface_smoothness": smoothness,
            "surface_orientation_barrier": orientation_barrier,
            "weak_symmetry": symmetry,
        }
    )
    term_slices = {}
    residual_parts = []
    cursor = 0
    for name, residual in named_residuals.items():
        residual_array = np.asarray(residual, dtype=np.float64).reshape(-1)
        residual_parts.append(residual_array)
        term_slices[name] = slice(cursor, cursor + len(residual_array))
        cursor += len(residual_array)
    residual_vector = np.concatenate(residual_parts)
    return _ObjectiveCoreEvaluation(
        candidate_vertices=candidate_vertices,
        residuals=residual_vector,
        term_slices=MappingProxyType(term_slices),
        term_residuals=MappingProxyType(named_residuals),
        soft_terms=MappingProxyType(soft_terms),
        effective_counts=MappingProxyType(effective_counts),
        sample_counts=MappingProxyType(sample_counts),
        confidence_sums=MappingProxyType(confidence_sums),
        per_view_effective_counts=MappingProxyType(
            per_view_effective_counts
        ),
        per_view_sample_counts=MappingProxyType(per_view_sample_counts),
        per_view_confidence_sums=MappingProxyType(
            per_view_confidence_sums
        ),
        symmetry_factors=MappingProxyType(symmetry_factors),
        soft_minimum_depths=soft_minimum_depths,
        orientation_signed_area_ratios=orientation_ratios,
        orientation_actual_minimum_signed_area_ratio=(
            orientation_actual_minimum
        ),
    )


def evaluate_multiview_nasal_objective_residuals(
    theta: np.ndarray,
    context: MultiviewNasalObjectiveContext,
    config: Optional[MultiviewNasalObjectiveConfig] = None,
) -> MultiviewNasalResidualEvaluation:
    """Return the minimal immutable residual output for iterative solvers."""
    core = _evaluate_multiview_nasal_objective_core(
        theta,
        context,
        config,
        include_diagnostics=False,
    )
    return MultiviewNasalResidualEvaluation(
        residuals=core.residuals,
        term_slices=core.term_slices,
        term_residuals=core.term_residuals,
    )


def evaluate_multiview_nasal_objective(
    theta: np.ndarray,
    context: MultiviewNasalObjectiveContext,
    config: Optional[MultiviewNasalObjectiveConfig] = None,
) -> MultiviewNasalObjectiveResult:
    """Evaluate unified residuals and construct immutable diagnostics."""
    limits = _objective_config(config)
    core = _evaluate_multiview_nasal_objective_core(
        theta,
        context,
        limits,
        include_diagnostics=True,
    )
    candidate = CandidateNasalMesh._from_validated(
        core.candidate_vertices,
        context.projection_context.faces,
        reuse_faces=True,
    )
    soft_projection_terms = tuple(
        SoftProfileNasalSlots(
            name=term.name,
            semantic_view=term.semantic_view,
            boundary_name=term.boundary_name,
            pixel_xy=core.soft_terms[term.name].pixel_xy,
            depth=core.soft_terms[term.name].depth,
            confidence=term.confidence,
            target_slot_y=term.slot_y,
            support_vertex_indices=term.support_vertex_indices,
            soft_weights=core.soft_terms[term.name].soft_weights,
            front_depth=core.soft_terms[term.name].front_depth,
            soft_visibility=core.soft_terms[term.name].soft_visibility,
            direction_sign=term.direction_sign,
        )
        for term in context.image_terms
    )
    projection = MultiviewNasalSoftProjection(
        candidate_vertices=candidate.vertices,
        per_term=soft_projection_terms,
    )
    named_residuals = core.term_residuals
    term_slices = core.term_slices
    raw_costs = {
        name: float(0.5 * np.dot(residual, residual))
        for name, residual in named_residuals.items()
    }
    robust_costs = {
        name: _soft_l1_cost(
            residual,
            float(limits.robust_f_scale),
        )
        for name, residual in named_residuals.items()
    }
    total_raw = float(sum(raw_costs.values()))
    total_robust = float(sum(robust_costs.values()))
    orientation_ratios = core.orientation_signed_area_ratios
    orientation_quantiles = np.quantile(
        orientation_ratios,
        (0.01, 0.05, 0.50, 0.95, 0.99),
    )
    orientation_diagnostic_soft_aggregate = (
        _soft_minimum_orientation_ratio(
            orientation_ratios,
            float(limits.orientation_barrier_softmin_temperature),
        )
    )
    report_data = {
        "raw_costs": raw_costs,
        "robust_costs": robust_costs,
        "total_raw_cost": total_raw,
        "total_robust_cost": total_robust,
        "effective_observation_counts": core.effective_counts,
        "sample_counts": core.sample_counts,
        "effective_confidence_sums": core.confidence_sums,
        "per_view_effective_observation_counts": (
            core.per_view_effective_counts
        ),
        "per_view_sample_counts": core.per_view_sample_counts,
        "per_view_effective_confidence_sums": (
            core.per_view_confidence_sums
        ),
        "symmetry_evidence_factors": core.symmetry_factors,
        "parameter_ordering": context.parameter_ordering,
        "parameter_bounds": {
            "lower": tuple(
                float(value) for value in context.parameter_lower_bounds
            ),
            "upper": tuple(
                float(value) for value in context.parameter_upper_bounds
            ),
            "solver_requirement": "C3 must enforce these fixed bounds",
        },
        "term_slices": {
            name: (value.start, value.stop)
            for name, value in term_slices.items()
        },
        "robust_loss": limits.robust_loss,
        "robust_f_scale": float(limits.robust_f_scale),
        "robust_loss_specification": {
            "loss": limits.robust_loss,
            "f_scale": float(limits.robust_f_scale),
        },
        "flame_prior_standard_deviations": tuple(
            float(value)
            for value in context.flame_mode_standard_deviations
        ),
        "semantic_prior_standard_deviations": (
            limits.semantic_prior_standard_deviations
        ),
        "smoothness_edge_count": context.smoothness_edge_count,
        "surface_orientation_barrier": {
            "active_face_count": (
                context.orientation_active_face_count
            ),
            "total_mesh_face_count": len(
                context.projection_context.faces
            ),
            "min_signed_area_ratio": (
                core.orientation_actual_minimum_signed_area_ratio
            ),
            "actual_min_signed_area_ratio": (
                core.orientation_actual_minimum_signed_area_ratio
            ),
            "global_feasibility_signed_area_ratio": (
                core.orientation_actual_minimum_signed_area_ratio
            ),
            "max_signed_area_ratio": float(
                np.max(orientation_ratios)
            ),
            "signed_area_ratio_quantiles": {
                "p01": float(orientation_quantiles[0]),
                "p05": float(orientation_quantiles[1]),
                "p50": float(orientation_quantiles[2]),
                "p95": float(orientation_quantiles[3]),
                "p99": float(orientation_quantiles[4]),
            },
            "diagnostic_normalized_log_mean_exp_signed_area_ratio": (
                orientation_diagnostic_soft_aggregate
            ),
            "nonpositive_face_count": int(
                np.count_nonzero(orientation_ratios <= 0.0)
            ),
            "faces_below_margin_count": int(
                np.count_nonzero(
                    orientation_ratios
                    < float(limits.orientation_barrier_margin)
                )
            ),
            "per_face_residual_count": (
                context.orientation_active_face_count
            ),
            "global_residual_count": 1,
            "weight": float(limits.orientation_barrier_weight),
            "margin": float(limits.orientation_barrier_margin),
            "scale": float(limits.orientation_barrier_scale),
            "softmin_temperature": float(
                limits.orientation_barrier_softmin_temperature
            ),
            "baseline_local_triangle_quality_threshold": (
                _ORIENTATION_BASELINE_LOCAL_QUALITY_THRESHOLD
            ),
        },
        "depth_support_vertex_count": len(
            context.depth_support_vertex_indices
        ),
        "profile_vertical_sigma_px": float(
            limits.profile_vertical_sigma_px
        ),
        "profile_softmax_temperature_px": float(
            limits.profile_softmax_temperature_px
        ),
        "profile_visibility_spatial_sigma_px": float(
            limits.profile_visibility_spatial_sigma_px
        ),
        "profile_front_depth_temperature": float(
            limits.profile_front_depth_temperature
        ),
        "profile_depth_visibility_scale": float(
            limits.profile_depth_visibility_scale
        ),
        "profile_depth_validity_scale": float(
            limits.profile_depth_validity_scale
        ),
        "soft_profile_summaries": {
            term.name: {
                "front_depth_min": float(
                    np.min(core.soft_terms[term.name].front_depth)
                ),
                "front_depth_max": float(
                    np.max(core.soft_terms[term.name].front_depth)
                ),
                "front_depth_mean": float(
                    np.mean(core.soft_terms[term.name].front_depth)
                ),
                "visibility_min": float(
                    np.min(core.soft_terms[term.name].soft_visibility)
                ),
                "visibility_max": float(
                    np.max(core.soft_terms[term.name].soft_visibility)
                ),
                "visibility_mean": float(
                    np.mean(core.soft_terms[term.name].soft_visibility)
                ),
            }
            for term in context.image_terms
        },
        "depth_softplus_scale": float(limits.depth_softplus_scale),
        "depth_barrier_weight": float(limits.depth_barrier_weight),
        "depth_barrier_scale": float(limits.depth_barrier_scale),
        "depth_barrier_softmin_temperature": float(
            limits.depth_barrier_softmin_temperature
        ),
        "soft_minimum_depths": {
            view: float(value)
            for view, value in zip(
                NASAL_VIEWS,
                core.soft_minimum_depths,
            )
        },
        "fixed_sampling_provenance": (
            "continuous_soft_semantic_profile_midpoint_quantile_slots"
        ),
    }
    return MultiviewNasalObjectiveResult(
        residuals=core.residuals,
        term_slices=term_slices,
        term_residuals=named_residuals,
        raw_costs=raw_costs,
        robust_costs=robust_costs,
        total_raw_cost=total_raw,
        total_robust_cost=total_robust,
        candidate=candidate,
        projection=projection,
        effective_observation_counts=core.effective_counts,
        sample_counts=core.sample_counts,
        effective_confidence_sums=core.confidence_sums,
        per_view_effective_observation_counts=(
            core.per_view_effective_counts
        ),
        per_view_sample_counts=core.per_view_sample_counts,
        per_view_effective_confidence_sums=(
            core.per_view_confidence_sums
        ),
        symmetry_evidence_factors=core.symmetry_factors,
        parameter_ordering=context.parameter_ordering,
        robust_loss=limits.robust_loss,
        robust_f_scale=float(limits.robust_f_scale),
        report_data=report_data,
    )

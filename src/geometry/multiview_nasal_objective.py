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
    "MultiviewNasalProjection",
    "MultiviewNasalSamplingConfig",
    "PreparedNasalProjectionContext",
    "ProjectedNasalSamples",
    "build_candidate_nasal_mesh",
    "build_candidate_nasal_mesh_prepared",
    "evaluate_multiview_nasal_objective",
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
        if semantic_view == "front":
            front_groups = _prepared_front_edge_groups(
                prepared,
                vertex_projection.pixel_xy,
            )
            edge_groups = [
                (
                    edges,
                    int(limits.front_samples_per_region),
                    source_label,
                    boundary_name,
                )
                for source_label, boundary_name, edges in front_groups
            ]
        else:
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
            edge_groups = [
                (
                    ordered_edges,
                    int(limits.side_samples_per_view),
                    None,
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
    "flame_prior",
    "semantic_prior",
    "surface_smoothness",
    "weak_symmetry",
)
_ASYMMETRY_MODE_NAMES = (
    "alar_width_asymmetry",
    "alar_depth_asymmetry",
)


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
    """Frozen subject-independent residual weights and robust-loss scales."""

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
    image_normalization_epsilon: float = 1e-12
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
        ):
            _finite_real(name, getattr(self, name))
        for name in (
            "smoothness_scale",
            "image_normalization_epsilon",
            "robust_f_scale",
        ):
            _finite_real(name, getattr(self, name), strictly_positive=True)
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
    source_vertex_indices: np.ndarray
    source_weights: np.ndarray
    source_labels: Tuple[str, ...]
    distance_field: np.ndarray
    confidence: np.ndarray
    work_size: Tuple[int, int]
    roi_work_xyxy: Tuple[float, float, float, float]

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
        indices = np.asarray(self.source_vertex_indices)
        weights = np.asarray(self.source_weights, dtype=np.float64)
        labels = tuple(str(value) for value in self.source_labels)
        if (
            indices.ndim != 2
            or indices.shape[1:] != (2,)
            or not len(indices)
            or not np.issubdtype(indices.dtype, np.integer)
        ):
            raise ValueError(
                "objective image source indices must have integer shape (N, 2)"
            )
        if (
            weights.shape != indices.shape
            or not np.isfinite(weights).all()
            or np.any(weights < 0.0)
            or not np.allclose(
                weights.sum(axis=1),
                1.0,
                atol=1e-12,
                rtol=0.0,
            )
        ):
            raise ValueError("objective image source weights are invalid")
        if len(labels) != len(indices) or set(labels) - set(_REGION_NAMES):
            raise ValueError("objective image source labels are invalid")
        distance = np.asarray(self.distance_field, dtype=np.float64)
        confidence = np.asarray(self.confidence, dtype=np.float64)
        work_size = tuple(int(value) for value in self.work_size)
        roi = tuple(float(value) for value in self.roi_work_xyxy)
        if (
            len(work_size) != 2
            or min(work_size) < 1
            or distance.shape != (work_size[1], work_size[0])
            or confidence.shape != (len(indices),)
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
            "source_vertex_indices",
            _readonly_array(indices, np.int64),
        )
        object.__setattr__(
            self,
            "source_weights",
            _readonly_array(weights, np.float64),
        )
        object.__setattr__(self, "source_labels", labels)
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

    @property
    def sample_count(self) -> int:
        return len(self.source_vertex_indices)


@dataclass(frozen=True)
class MultiviewNasalObjectiveContext:
    """Static, deeply immutable state reused by objective evaluations."""

    baseline_vertices: np.ndarray
    observable_vertex_basis: np.ndarray
    projection_context: PreparedNasalProjectionContext
    image_terms: Tuple[_PreparedObjectiveImageTerm, ...]
    flame_mode_standard_deviations: np.ndarray
    smoothness_centers: np.ndarray
    smoothness_neighbors: np.ndarray
    smoothness_neighbor_offsets: np.ndarray
    smoothness_edge_count: int
    parameter_ordering: Tuple[str, ...]

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
            if (
                np.any(term.source_vertex_indices < 0)
                or np.any(
                    term.source_vertex_indices >= prepared.vertex_count
                )
            ):
                raise ValueError(
                    f"objective image term {term.name} has out-of-range "
                    "source indices"
                )
            for label, edge in zip(
                term.source_labels,
                term.source_vertex_indices,
            ):
                if not np.all(prepared.region_masks[label][edge]):
                    raise ValueError(
                        f"objective image term {term.name} has invalid "
                        "semantic provenance"
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
        ordering = tuple(str(value) for value in self.parameter_ordering)
        expected_ordering = tuple(
            f"observable_flame_{index}" for index in range(rank)
        ) + tuple(NASAL_SEMANTIC_MODE_NAMES)
        if ordering != expected_ordering:
            raise ValueError(
                "parameter_ordering must list observable modes then "
                "canonical semantic modes"
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
        object.__setattr__(self, "parameter_ordering", ordering)

    @property
    def observable_rank(self) -> int:
        return self.observable_vertex_basis.shape[2]

    @property
    def parameter_count(self) -> int:
        return self.observable_rank + len(NASAL_SEMANTIC_MODE_NAMES)

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
    projection: MultiviewNasalProjection
    effective_observation_counts: Mapping[str, int]
    sample_counts: Mapping[str, int]
    effective_confidence_sums: Mapping[str, float]
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
        if not isinstance(self.projection, MultiviewNasalProjection):
            raise ValueError("projection must be a MultiviewNasalProjection")
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
    baseline_candidate = build_candidate_nasal_mesh_prepared(
        baseline,
        observable_flame,
        prepared,
        np.zeros(rank, dtype=np.float64),
        np.zeros(len(NASAL_SEMANTIC_MODE_NAMES), dtype=np.float64),
    )
    baseline_projection = project_multiview_nasal_boundaries_prepared(
        baseline_candidate,
        prepared,
    )
    image_terms = []
    projection_by_view = baseline_projection.by_view
    for name, semantic_view, boundary_name in _OBJECTIVE_IMAGE_TERM_SPECS:
        samples = projection_by_view[semantic_view]
        selected = np.asarray(
            [
                value == boundary_name
                for value in samples.boundary_names
            ],
            dtype=bool,
        )
        if not np.any(selected):
            raise ValueError(
                f"baseline projection has no fixed slots for {name}"
            )
        observation = observations.by_view[semantic_view]
        image_terms.append(
            _PreparedObjectiveImageTerm(
                name=name,
                semantic_view=semantic_view,
                boundary_name=boundary_name,
                source_vertex_indices=samples.source_vertex_indices[
                    selected
                ],
                source_weights=samples.source_weights[selected],
                source_labels=tuple(
                    label
                    for label, keep in zip(
                        samples.source_labels,
                        selected,
                    )
                    if keep
                ),
                distance_field=observation.distance_fields[boundary_name],
                confidence=samples.confidence[selected],
                work_size=tuple(observation.work_size),
                roi_work_xyxy=tuple(observation.roi_work_xyxy),
            )
        )
    centers, neighbors, offsets, edge_count = _smoothness_topology(prepared)
    ordering = tuple(
        f"observable_flame_{index}" for index in range(rank)
    ) + tuple(NASAL_SEMANTIC_MODE_NAMES)
    return MultiviewNasalObjectiveContext(
        baseline_vertices=baseline,
        observable_vertex_basis=observable_basis,
        projection_context=prepared,
        image_terms=tuple(image_terms),
        flame_mode_standard_deviations=flame_standard_deviations,
        smoothness_centers=centers,
        smoothness_neighbors=neighbors,
        smoothness_neighbor_offsets=offsets,
        smoothness_edge_count=edge_count,
        parameter_ordering=ordering,
    )


def _bilinear_sample_field(
    field: np.ndarray,
    pixels: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
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
    return values, inside


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


def evaluate_multiview_nasal_objective(
    theta: np.ndarray,
    context: MultiviewNasalObjectiveContext,
    config: Optional[MultiviewNasalObjectiveConfig] = None,
) -> MultiviewNasalObjectiveResult:
    """Evaluate one fixed-length unified multiview nasal residual vector."""
    if not isinstance(context, MultiviewNasalObjectiveContext):
        raise ValueError(
            "context must be a MultiviewNasalObjectiveContext"
        )
    limits = MultiviewNasalObjectiveConfig() if config is None else config
    if not isinstance(limits, MultiviewNasalObjectiveConfig):
        raise ValueError("config must be a MultiviewNasalObjectiveConfig")
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
    candidate = CandidateNasalMesh._from_validated(
        candidate_vertices,
        context.projection_context.faces,
        reuse_faces=True,
    )

    image_residuals = {}
    projected_term_data = {}
    confidence_sums = {}
    effective_counts = {}
    sample_counts = {}
    reliability = {}
    for term in context.image_terms:
        model_points = np.sum(
            candidate.vertices[term.source_vertex_indices]
            * term.source_weights[:, :, None],
            axis=1,
        )
        view_index = NASAL_VIEWS.index(term.semantic_view)
        view = context.projection_context.views[view_index]
        projected = project_points_strict(
            model_points,
            view.K,
            view.R_model_to_camera,
            view.t_model_to_camera,
            epsilon=float(
                context.projection_context.config.min_depth
            ),
        )
        pixels = np.asarray(projected.pixel_xy, dtype=np.float64)
        distances, inside_image = _bilinear_sample_field(
            term.distance_field,
            pixels,
        )
        x0, y0, x1, y1 = term.roi_work_xyxy
        inside_roi = (
            (pixels[:, 0] >= x0)
            & (pixels[:, 0] < x1)
            & (pixels[:, 1] >= y0)
            & (pixels[:, 1] < y1)
        )
        valid = inside_image & inside_roi
        confidence = np.where(valid, term.confidence, 0.0)
        confidence_sum = float(np.sum(confidence))
        denominator = np.sqrt(
            confidence_sum
            + float(limits.image_normalization_epsilon)
        )
        term_weight = (
            float(limits.front_image_weight)
            if term.semantic_view == "front"
            else float(limits.side_image_weight)
        )
        residual = (
            term_weight
            * np.sqrt(confidence)
            * distances
            / denominator
        )
        image_residuals[term.name] = residual
        confidence_sums[term.name] = confidence_sum
        effective_counts[term.name] = int(
            np.count_nonzero(confidence > 0.0)
        )
        sample_counts[term.name] = term.sample_count
        reliability[term.name] = (
            confidence_sum / float(term.sample_count)
        )
        projected_term_data[term.name] = (
            model_points,
            pixels,
            projected.depth,
            confidence,
        )

    per_view = []
    for semantic_view in NASAL_VIEWS:
        matching_terms = [
            term
            for term in context.image_terms
            if term.semantic_view == semantic_view
        ]
        model_points = np.vstack(
            [projected_term_data[term.name][0] for term in matching_terms]
        )
        pixels = np.vstack(
            [projected_term_data[term.name][1] for term in matching_terms]
        )
        depth = np.concatenate(
            [projected_term_data[term.name][2] for term in matching_terms]
        )
        confidence = np.concatenate(
            [projected_term_data[term.name][3] for term in matching_terms]
        )
        indices = np.vstack(
            [term.source_vertex_indices for term in matching_terms]
        )
        weights = np.vstack(
            [term.source_weights for term in matching_terms]
        )
        labels = tuple(
            label
            for term in matching_terms
            for label in term.source_labels
        )
        boundaries = tuple(
            term.boundary_name
            for term in matching_terms
            for _ in range(term.sample_count)
        )
        per_view.append(
            ProjectedNasalSamples(
                semantic_view=semantic_view,
                pixel_xy=pixels,
                model_points=model_points,
                source_vertex_indices=indices,
                source_weights=weights,
                confidence=confidence,
                source_labels=labels,
                boundary_names=boundaries,
                visible=np.ones(len(pixels), dtype=bool),
                depth=depth,
            )
        )
    projection = MultiviewNasalProjection._from_prepared(
        candidate,
        tuple(per_view),
        context.projection_context,
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
    displacement = candidate.vertices - context.baseline_vertices
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
            "flame_prior": flame_prior,
            "semantic_prior": semantic_prior,
            "surface_smoothness": smoothness,
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
    report_data = {
        "raw_costs": raw_costs,
        "robust_costs": robust_costs,
        "total_raw_cost": total_raw,
        "total_robust_cost": total_robust,
        "effective_observation_counts": effective_counts,
        "sample_counts": sample_counts,
        "effective_confidence_sums": confidence_sums,
        "symmetry_evidence_factors": symmetry_factors,
        "parameter_ordering": context.parameter_ordering,
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
        "fixed_sampling_provenance": "baseline_edge_slots",
    }
    return MultiviewNasalObjectiveResult(
        residuals=residual_vector,
        term_slices=term_slices,
        term_residuals=named_residuals,
        raw_costs=raw_costs,
        robust_costs=robust_costs,
        total_raw_cost=total_raw,
        total_robust_cost=total_robust,
        candidate=candidate,
        projection=projection,
        effective_observation_counts=effective_counts,
        sample_counts=sample_counts,
        effective_confidence_sums=confidence_sums,
        symmetry_evidence_factors=symmetry_factors,
        parameter_ordering=context.parameter_ordering,
        robust_loss=limits.robust_loss,
        robust_f_scale=float(limits.robust_f_scale),
        report_data=report_data,
    )

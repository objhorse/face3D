"""One-shot robust image offsets for protected nasal surface evidence."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import numpy as np


NASAL_REGISTRATION_VIEWS = ("front", "subject-left", "subject-right")
PROTECTED_REGISTRATION_REGIONS = frozenset(
    {"bridge", "peri_nasal_skin", "subject_left_peri_nasal", "subject_right_peri_nasal"}
)


def _readonly(value: Any, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _point(name: str, value: Any) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must have finite shape (2,)")
    return _readonly(point, np.float64)


def _point3(name: str, value: Any) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (3,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must have finite shape (3,)")
    return _readonly(point, np.float64)


@dataclass(frozen=True)
class BaselineNasalRegistrationSurface:
    vertices: np.ndarray
    faces: np.ndarray
    protected_face_indices_by_region: Mapping[str, np.ndarray]
    rig: Any
    provenance_by_view: Mapping[str, Any]
    model_to_front_rotation: np.ndarray
    model_to_front_translation: np.ndarray
    rig_view_by_semantic: Mapping[str, str] = field(
        default_factory=lambda: {
            "front": "front",
            "subject-left": "left",
            "subject-right": "right",
        }
    )
    projection_matrices_by_view: Mapping[str, np.ndarray] = field(init=False)

    def __post_init__(self) -> None:
        vertices = np.asarray(self.vertices, dtype=np.float64)
        faces = np.asarray(self.faces, dtype=np.int64)
        if (
            vertices.ndim != 2
            or vertices.shape[1] != 3
            or len(vertices) == 0
            or not np.isfinite(vertices).all()
        ):
            raise ValueError("vertices must have finite shape (N, 3)")
        if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
            raise ValueError("faces must have shape (F, 3)")
        if np.any(faces < 0) or np.any(faces >= len(vertices)):
            raise ValueError("faces contain invalid vertex indices")
        if set(self.protected_face_indices_by_region) != set(
            PROTECTED_REGISTRATION_REGIONS
        ):
            raise ValueError("protected face mapping must contain every region")
        protected: dict[str, np.ndarray] = {}
        for region in sorted(PROTECTED_REGISTRATION_REGIONS):
            indices = np.asarray(
                self.protected_face_indices_by_region[region],
                dtype=np.int64,
            ).reshape(-1)
            if np.any(indices < 0) or np.any(indices >= len(faces)):
                raise ValueError(f"protected faces for {region} are out of range")
            protected[region] = _readonly(np.unique(indices), np.int64)
        from src.geometry.nasal_texture_observations import (
            NasalCoordinateProvenance,
            _validate_rig_view_mapping,
            _work_intrinsics_from_provenance,
        )
        from src.geometry.profile_triangulation import ProfileRig

        if not isinstance(self.rig, ProfileRig):
            raise ValueError("rig must be a validated ProfileRig")
        provenance = dict(self.provenance_by_view)
        if set(provenance) != set(NASAL_REGISTRATION_VIEWS):
            raise ValueError("coordinate provenance must contain every semantic view")
        for view, item in provenance.items():
            if not isinstance(item, NasalCoordinateProvenance):
                raise ValueError("invalid coordinate provenance")
            if item.semantic_view != view:
                raise ValueError("coordinate provenance view mismatch")
        view_map = {str(key): str(value) for key, value in self.rig_view_by_semantic.items()}
        if set(view_map) != set(NASAL_REGISTRATION_VIEWS):
            raise ValueError("rig view mapping must contain every semantic view")
        _validate_rig_view_mapping(self.rig, view_map)
        model_rotation = np.asarray(self.model_to_front_rotation, dtype=np.float64)
        model_translation = np.asarray(self.model_to_front_translation, dtype=np.float64)
        if (
            model_rotation.shape != (3, 3)
            or not np.isfinite(model_rotation).all()
            or not np.allclose(model_rotation.T @ model_rotation, np.eye(3), atol=1e-6)
            or not np.isclose(np.linalg.det(model_rotation), 1.0, atol=1e-6)
        ):
            raise ValueError("model_to_front_rotation must be a proper rotation")
        if model_translation.shape != (3,) or not np.isfinite(model_translation).all():
            raise ValueError("model_to_front_translation must have finite shape (3,)")

        front_camera = self.rig.cameras_by_view[view_map["front"]]
        front_rotation = np.asarray(front_camera.R_rig_to_camera, dtype=np.float64)
        front_translation = np.asarray(front_camera.t_rig_to_camera, dtype=np.float64)
        projections: dict[str, np.ndarray] = {}
        for view in NASAL_REGISTRATION_VIEWS:
            camera = self.rig.cameras_by_view[view_map[view]]
            target_rotation = np.asarray(camera.R_rig_to_camera, dtype=np.float64)
            target_translation = np.asarray(camera.t_rig_to_camera, dtype=np.float64)
            target_from_front_rotation = target_rotation @ front_rotation.T
            target_from_front_translation = (
                target_translation
                - target_from_front_rotation @ front_translation
            )
            model_to_target_rotation = target_from_front_rotation @ model_rotation
            model_to_target_translation = (
                target_from_front_rotation @ model_translation
                + target_from_front_translation
            )
            work_intrinsics = _work_intrinsics_from_provenance(
                camera,
                provenance[view],
            )
            projection = work_intrinsics @ np.column_stack(
                (model_to_target_rotation, model_to_target_translation)
            )
            if projection.shape != (3, 4) or not np.isfinite(projection).all():
                raise ValueError(f"derived projection matrix for {view} is invalid")
            projections[view] = _readonly(projection, np.float64)
        object.__setattr__(self, "vertices", _readonly(vertices, np.float64))
        object.__setattr__(self, "faces", _readonly(faces, np.int64))
        object.__setattr__(
            self,
            "protected_face_indices_by_region",
            MappingProxyType(protected),
        )
        object.__setattr__(
            self,
            "projection_matrices_by_view",
            MappingProxyType(projections),
        )
        object.__setattr__(self, "provenance_by_view", MappingProxyType(provenance))
        object.__setattr__(self, "model_to_front_rotation", _readonly(model_rotation))
        object.__setattr__(self, "model_to_front_translation", _readonly(model_translation))
        object.__setattr__(self, "rig_view_by_semantic", MappingProxyType(view_map))

    def baseline_point(self, sample: "ViewRegistrationSample") -> np.ndarray | None:
        face_index = int(sample.baseline_face_index)
        if face_index < 0 or face_index >= len(self.faces):
            return None
        protected = self.protected_face_indices_by_region.get(sample.semantic_region)
        if protected is None or not np.any(protected == face_index):
            return None
        triangle = self.vertices[self.faces[face_index]]
        return np.sum(
            triangle * sample.baseline_bary_coords[:, None],
            axis=0,
        )

    def verified_surface_distance(self, sample: "ViewRegistrationSample") -> float:
        baseline_point = self.baseline_point(sample)
        if baseline_point is None:
            return float("inf")
        return float(np.linalg.norm(sample.matched_point_3d - baseline_point))

    def verified_projected_pixel(self, sample: "ViewRegistrationSample") -> np.ndarray:
        baseline_point = self.baseline_point(sample)
        if baseline_point is None:
            return np.full(2, np.nan, dtype=np.float64)
        homogeneous = self.projection_matrices_by_view[sample.semantic_view] @ np.append(
            baseline_point,
            1.0,
        )
        if not np.isfinite(homogeneous).all() or abs(float(homogeneous[2])) <= 1e-12:
            return np.full(2, np.nan, dtype=np.float64)
        return homogeneous[:2] / homogeneous[2]


@dataclass(frozen=True)
class NasalViewRegistrationConfig:
    min_samples_per_view: int = 6
    min_semantic_regions_per_view: int = 2
    min_sample_confidence: float = 0.5
    max_offset_px: float = 4.0
    robust_floor_px: float = 0.25
    outlier_sigma: float = 3.5
    max_surface_distance_m: float = 0.01
    zero_prior_weight: float = 0.05
    max_residual_p90_px: float = 1.0
    min_registration_confidence: float = 0.1
    max_baseline_projection_error_px: float = 0.25
    zero_mean_gauge: bool = True

    def __post_init__(self) -> None:
        for name in ("min_samples_per_view", "min_semantic_regions_per_view"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            if int(value) < 1:
                raise ValueError(f"{name} must be positive")
        if not 0.0 < float(self.min_sample_confidence) <= 1.0:
            raise ValueError("min_sample_confidence must lie in (0, 1]")
        for name in (
            "max_offset_px",
            "robust_floor_px",
            "outlier_sigma",
            "max_surface_distance_m",
            "zero_prior_weight",
            "max_residual_p90_px",
            "min_registration_confidence",
            "max_baseline_projection_error_px",
        ):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if not isinstance(self.zero_mean_gauge, (bool, np.bool_)):
            raise ValueError("zero_mean_gauge must be boolean")
        if float(self.min_registration_confidence) > 1.0:
            raise ValueError("min_registration_confidence must not exceed 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "min_samples_per_view": int(self.min_samples_per_view),
            "min_semantic_regions_per_view": int(
                self.min_semantic_regions_per_view
            ),
            "min_sample_confidence": float(self.min_sample_confidence),
            "max_offset_px": float(self.max_offset_px),
            "robust_floor_px": float(self.robust_floor_px),
            "outlier_sigma": float(self.outlier_sigma),
            "max_surface_distance_m": float(self.max_surface_distance_m),
            "zero_prior_weight": float(self.zero_prior_weight),
            "max_residual_p90_px": float(self.max_residual_p90_px),
            "min_registration_confidence": float(
                self.min_registration_confidence
            ),
            "max_baseline_projection_error_px": float(
                self.max_baseline_projection_error_px
            ),
            "zero_mean_gauge": bool(self.zero_mean_gauge),
        }


@dataclass(frozen=True)
class ViewRegistrationSample:
    semantic_view: str
    projected_pixel: np.ndarray
    observed_pixel: np.ndarray
    confidence: float
    semantic_region: str
    source: str
    baseline_face_index: int
    baseline_bary_coords: np.ndarray
    matched_point_3d: np.ndarray

    def __post_init__(self) -> None:
        view = str(self.semantic_view)
        if view not in NASAL_REGISTRATION_VIEWS:
            raise ValueError(f"unknown semantic_view: {view}")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        region = str(self.semantic_region).strip()
        source = str(self.source).strip()
        if not region:
            raise ValueError("semantic_region must not be empty")
        if not source:
            raise ValueError("source must not be empty")
        if isinstance(self.baseline_face_index, bool) or not isinstance(
            self.baseline_face_index, (int, np.integer)
        ):
            raise ValueError("baseline_face_index must be an integer")
        face_index = int(self.baseline_face_index)
        if face_index < 0:
            raise ValueError("baseline_face_index must be non-negative")
        barycentric = np.asarray(self.baseline_bary_coords, dtype=np.float64)
        if (
            barycentric.shape != (3,)
            or not np.isfinite(barycentric).all()
            or np.any(barycentric < -1e-8)
            or not np.isclose(float(np.sum(barycentric)), 1.0, atol=1e-8)
        ):
            raise ValueError("baseline_bary_coords must be valid barycentric weights")
        object.__setattr__(self, "semantic_view", view)
        object.__setattr__(
            self,
            "projected_pixel",
            _point("projected_pixel", self.projected_pixel),
        )
        object.__setattr__(
            self,
            "observed_pixel",
            _point("observed_pixel", self.observed_pixel),
        )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "semantic_region", region)
        object.__setattr__(self, "source", source)
        object.__setattr__(self, "baseline_face_index", face_index)
        object.__setattr__(
            self,
            "baseline_bary_coords",
            _readonly(barycentric, np.float64),
        )
        object.__setattr__(
            self,
            "matched_point_3d",
            _point3("matched_point_3d", self.matched_point_3d),
        )

    @property
    def residual(self) -> np.ndarray:
        return _readonly(self.observed_pixel - self.projected_pixel, np.float64)

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_view": self.semantic_view,
            "projected_pixel": self.projected_pixel.tolist(),
            "observed_pixel": self.observed_pixel.tolist(),
            "confidence": self.confidence,
            "semantic_region": self.semantic_region,
            "source": self.source,
            "baseline_face_index": self.baseline_face_index,
            "baseline_bary_coords": self.baseline_bary_coords.tolist(),
            "matched_point_3d": self.matched_point_3d.tolist(),
        }


def _frozen_array_mapping(values: Mapping[str, Any]) -> Mapping[str, np.ndarray]:
    return MappingProxyType(
        {
            view: _readonly(values[view], np.float64)
            for view in NASAL_REGISTRATION_VIEWS
        }
    )


def _frozen_scalar_mapping(
    values: Mapping[str, Any],
    cast,
) -> Mapping[str, Any]:
    return MappingProxyType(
        {view: cast(values[view]) for view in NASAL_REGISTRATION_VIEWS}
    )


@dataclass(frozen=True)
class FixedNasalViewRegistration:
    offsets_by_view: Mapping[str, np.ndarray]
    confidence_by_view: Mapping[str, float]
    sample_counts_by_view: Mapping[str, int]
    inlier_counts_by_view: Mapping[str, int]
    semantic_region_counts_by_view: Mapping[str, int]
    residual_p90_px_by_view: Mapping[str, float]
    residual_quantiles_px_by_view: Mapping[str, np.ndarray]
    offset_confidence_interval95_px_by_view: Mapping[str, np.ndarray]
    accepted_by_view: Mapping[str, bool]
    config: NasalViewRegistrationConfig

    def __post_init__(self) -> None:
        expected = set(NASAL_REGISTRATION_VIEWS)
        for name in (
            "offsets_by_view",
            "confidence_by_view",
            "sample_counts_by_view",
            "inlier_counts_by_view",
            "semantic_region_counts_by_view",
            "residual_p90_px_by_view",
            "residual_quantiles_px_by_view",
            "offset_confidence_interval95_px_by_view",
            "accepted_by_view",
        ):
            if set(getattr(self, name)) != expected:
                raise ValueError(f"{name} must contain all semantic views")
        offsets = _frozen_array_mapping(self.offsets_by_view)
        if any(value.shape != (2,) or not np.isfinite(value).all() for value in offsets.values()):
            raise ValueError("view offsets must have finite shape (2,)")
        confidence = _frozen_scalar_mapping(self.confidence_by_view, float)
        counts = _frozen_scalar_mapping(self.sample_counts_by_view, int)
        inlier_counts = _frozen_scalar_mapping(self.inlier_counts_by_view, int)
        region_counts = _frozen_scalar_mapping(
            self.semantic_region_counts_by_view,
            int,
        )
        p90 = _frozen_scalar_mapping(self.residual_p90_px_by_view, float)
        quantiles = _frozen_array_mapping(self.residual_quantiles_px_by_view)
        intervals = _frozen_array_mapping(
            self.offset_confidence_interval95_px_by_view
        )
        accepted = _frozen_scalar_mapping(self.accepted_by_view, bool)
        if any(not 0.0 <= value <= 1.0 for value in confidence.values()):
            raise ValueError("registration confidence must lie in [0, 1]")
        if any(value < 0 for value in counts.values()) or any(
            value < 0 or value > counts[view]
            for view, value in inlier_counts.items()
        ):
            raise ValueError("registration sample counts must be non-negative")
        if any(value < 0 for value in region_counts.values()):
            raise ValueError("semantic region counts must be non-negative")
        if any(
            value.shape != (3,) or not np.isfinite(value).all()
            for value in quantiles.values()
        ):
            raise ValueError("residual quantiles must have finite shape (3,)")
        if any(
            value.shape != (2, 2) or not np.isfinite(value).all()
            for value in intervals.values()
        ):
            raise ValueError("offset confidence intervals must have finite shape (2, 2)")
        if any(not np.isfinite(value) or value < 0.0 for value in p90.values()):
            raise ValueError("registration residuals must be finite and non-negative")
        if not isinstance(self.config, NasalViewRegistrationConfig):
            raise ValueError("config must be a NasalViewRegistrationConfig")
        for view in NASAL_REGISTRATION_VIEWS:
            if not accepted[view] and (
                np.linalg.norm(offsets[view]) > 1e-12
                or confidence[view] != 0.0
            ):
                raise ValueError("rejected registration views must be zeroed")
            if accepted[view] and (
                confidence[view] < float(self.config.min_registration_confidence)
                or np.linalg.norm(offsets[view]) > float(self.config.max_offset_px)
                or p90[view] > float(self.config.max_residual_p90_px)
                or inlier_counts[view] < int(self.config.min_samples_per_view)
                or region_counts[view]
                < int(self.config.min_semantic_regions_per_view)
            ):
                raise ValueError("accepted registration view violates configured gates")
        object.__setattr__(self, "offsets_by_view", offsets)
        object.__setattr__(self, "confidence_by_view", confidence)
        object.__setattr__(self, "sample_counts_by_view", counts)
        object.__setattr__(self, "inlier_counts_by_view", inlier_counts)
        object.__setattr__(
            self,
            "semantic_region_counts_by_view",
            region_counts,
        )
        object.__setattr__(self, "residual_p90_px_by_view", p90)
        object.__setattr__(self, "residual_quantiles_px_by_view", quantiles)
        object.__setattr__(
            self,
            "offset_confidence_interval95_px_by_view",
            intervals,
        )
        object.__setattr__(self, "accepted_by_view", accepted)

    @property
    def all_views_accepted(self) -> bool:
        return all(self.accepted_by_view.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "offsets_by_view": {
                view: value.tolist() for view, value in self.offsets_by_view.items()
            },
            "confidence_by_view": dict(self.confidence_by_view),
            "sample_counts_by_view": dict(self.sample_counts_by_view),
            "inlier_counts_by_view": dict(self.inlier_counts_by_view),
            "semantic_region_counts_by_view": dict(
                self.semantic_region_counts_by_view
            ),
            "residual_p90_px_by_view": dict(self.residual_p90_px_by_view),
            "residual_quantiles_px_by_view": {
                view: value.tolist()
                for view, value in self.residual_quantiles_px_by_view.items()
            },
            "offset_confidence_interval95_px_by_view": {
                view: value.tolist()
                for view, value in self.offset_confidence_interval95_px_by_view.items()
            },
            "accepted_by_view": dict(self.accepted_by_view),
            "all_views_accepted": self.all_views_accepted,
            "config": self.config.to_dict(),
        }


def _weighted_median(values: np.ndarray, weights: np.ndarray) -> float:
    order = np.argsort(values, kind="stable")
    sorted_values = values[order]
    sorted_weights = weights[order]
    threshold = 0.5 * float(np.sum(sorted_weights))
    index = int(np.searchsorted(np.cumsum(sorted_weights), threshold, side="left"))
    return float(sorted_values[min(index, len(sorted_values) - 1)])


def _robust_offset(
    residuals: np.ndarray,
    weights: np.ndarray,
    config: NasalViewRegistrationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    initial = np.asarray(
        [
            _weighted_median(residuals[:, axis], weights)
            for axis in range(2)
        ],
        dtype=np.float64,
    )
    distances = np.linalg.norm(residuals - initial, axis=1)
    median_distance = float(np.median(distances))
    mad = float(np.median(np.abs(distances - median_distance)))
    cutoff = max(
        float(config.robust_floor_px),
        median_distance + float(config.outlier_sigma) * 1.4826 * mad,
    )
    inliers = distances <= cutoff
    if not np.any(inliers):
        return initial, np.ones(len(residuals), dtype=bool)
    local_weights = weights[inliers]
    weighted_sum = np.sum(
        residuals[inliers] * local_weights[:, None],
        axis=0,
    )
    refined = weighted_sum / (
        float(np.sum(local_weights)) + float(config.zero_prior_weight)
    )
    return np.asarray(refined, dtype=np.float64), inliers


def _offset_diagnostics(
    residuals: np.ndarray,
    inliers: np.ndarray,
    offset: np.ndarray,
) -> tuple[float, np.ndarray, np.ndarray]:
    errors_xy = residuals[inliers] - offset
    distances = np.linalg.norm(errors_xy, axis=1)
    quantiles = np.percentile(distances, [50.0, 90.0, 99.0])
    coordinate_mad = np.median(
        np.abs(errors_xy - np.median(errors_xy, axis=0)),
        axis=0,
    )
    half_width = 1.96 * 1.4826 * coordinate_mad / np.sqrt(max(len(errors_xy), 1))
    interval = np.stack((offset - half_width, offset + half_width), axis=0)
    return float(quantiles[1]), quantiles.astype(np.float64), interval


def estimate_fixed_nasal_view_offsets(
    samples: Sequence[ViewRegistrationSample],
    baseline_surface: BaselineNasalRegistrationSurface,
    config: NasalViewRegistrationConfig | None = None,
) -> FixedNasalViewRegistration:
    """Estimate robust offsets once; no shape or camera variables are exposed."""
    cfg = config or NasalViewRegistrationConfig()
    records = tuple(samples)
    if not all(isinstance(value, ViewRegistrationSample) for value in records):
        raise ValueError("samples contain an invalid registration record")
    if not isinstance(baseline_surface, BaselineNasalRegistrationSurface):
        raise ValueError("baseline_surface must be verified baseline geometry")

    offsets = {view: np.zeros(2, dtype=np.float64) for view in NASAL_REGISTRATION_VIEWS}
    confidence = {view: 0.0 for view in NASAL_REGISTRATION_VIEWS}
    counts = {view: 0 for view in NASAL_REGISTRATION_VIEWS}
    inlier_counts = {view: 0 for view in NASAL_REGISTRATION_VIEWS}
    region_counts = {view: 0 for view in NASAL_REGISTRATION_VIEWS}
    p90 = {view: 0.0 for view in NASAL_REGISTRATION_VIEWS}
    quantiles = {view: np.zeros(3, dtype=np.float64) for view in NASAL_REGISTRATION_VIEWS}
    intervals = {
        view: np.zeros((2, 2), dtype=np.float64)
        for view in NASAL_REGISTRATION_VIEWS
    }
    accepted = {view: False for view in NASAL_REGISTRATION_VIEWS}
    fit_data: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    raw_offsets: dict[str, np.ndarray] = {}

    for view in NASAL_REGISTRATION_VIEWS:
        usable = []
        for sample in records:
            if (
                sample.semantic_view != view
                or sample.semantic_region not in PROTECTED_REGISTRATION_REGIONS
                or sample.source != "protected_surface_correspondence"
                or sample.confidence < float(cfg.min_sample_confidence)
                or baseline_surface.verified_surface_distance(sample)
                > float(cfg.max_surface_distance_m)
            ):
                continue
            verified_pixel = baseline_surface.verified_projected_pixel(sample)
            if (
                not np.isfinite(verified_pixel).all()
                or np.linalg.norm(verified_pixel - sample.projected_pixel)
                > float(cfg.max_baseline_projection_error_px)
            ):
                continue
            usable.append(sample)
        unique: dict[tuple[Any, ...], ViewRegistrationSample] = {}
        for sample in usable:
            anchor_key = (
                sample.semantic_region,
                sample.baseline_face_index,
                tuple(np.round(sample.baseline_bary_coords, 10)),
            )
            previous = unique.get(anchor_key)
            if previous is None or sample.confidence > previous.confidence:
                unique[anchor_key] = sample
        usable = tuple(unique[key] for key in sorted(unique, key=str))
        counts[view] = len(usable)
        if len(usable) < int(cfg.min_samples_per_view):
            continue
        residuals = np.stack([sample.residual for sample in usable], axis=0)
        weights = np.asarray([sample.confidence for sample in usable], dtype=np.float64)
        estimate, inliers = _robust_offset(residuals, weights, cfg)
        fit_data[view] = (residuals, inliers)
        inlier_counts[view] = int(np.count_nonzero(inliers))
        if inlier_counts[view] < int(cfg.min_samples_per_view):
            continue
        region_counts[view] = len(
            {
                sample.semantic_region
                for sample, is_inlier in zip(usable, inliers)
                if is_inlier
            }
        )
        if region_counts[view] < int(cfg.min_semantic_regions_per_view):
            continue
        residual_p90, view_quantiles, view_interval = _offset_diagnostics(
            residuals,
            inliers,
            estimate,
        )
        p90[view] = residual_p90
        quantiles[view] = view_quantiles
        intervals[view] = view_interval
        if np.linalg.norm(estimate) > float(cfg.max_offset_px):
            continue
        support_factor = min(
            1.0,
            inlier_counts[view] / (2.0 * int(cfg.min_samples_per_view)),
        )
        residual_factor = float(np.exp(-residual_p90 / max(cfg.robust_floor_px, 1e-6)))
        candidate_confidence = float(
            np.clip(support_factor * residual_factor, 0.0, 1.0)
        )
        if (
            residual_p90 <= float(cfg.max_residual_p90_px)
            and candidate_confidence >= float(cfg.min_registration_confidence)
        ):
            raw_offsets[view] = estimate
            confidence[view] = candidate_confidence
            accepted[view] = True

    active = {view for view in NASAL_REGISTRATION_VIEWS if accepted[view]}
    while active:
        if bool(cfg.zero_mean_gauge):
            gauge = np.mean(
                np.stack([raw_offsets[view] for view in sorted(active)], axis=0),
                axis=0,
            )
        else:
            gauge = np.zeros(2, dtype=np.float64)
        failed = set()
        for view in active:
            candidate = raw_offsets[view] - gauge
            residuals, inliers = fit_data[view]
            residual_p90, view_quantiles, view_interval = _offset_diagnostics(
                residuals,
                inliers,
                candidate,
            )
            support_factor = min(
                1.0,
                inlier_counts[view] / (2.0 * int(cfg.min_samples_per_view)),
            )
            candidate_confidence = float(
                np.clip(
                    support_factor
                    * np.exp(-residual_p90 / max(cfg.robust_floor_px, 1e-6)),
                    0.0,
                    1.0,
                )
            )
            p90[view] = residual_p90
            quantiles[view] = view_quantiles
            intervals[view] = view_interval
            if (
                np.linalg.norm(candidate) > float(cfg.max_offset_px)
                or
                residual_p90 > float(cfg.max_residual_p90_px)
                or candidate_confidence < float(cfg.min_registration_confidence)
            ):
                failed.add(view)
                continue
            offsets[view] = candidate
            confidence[view] = candidate_confidence
        if not failed:
            break
        for view in failed:
            accepted[view] = False
            offsets[view] = np.zeros(2, dtype=np.float64)
            confidence[view] = 0.0
        active -= failed

    return FixedNasalViewRegistration(
        offsets_by_view=offsets,
        confidence_by_view=confidence,
        sample_counts_by_view=counts,
        inlier_counts_by_view=inlier_counts,
        semantic_region_counts_by_view=region_counts,
        residual_p90_px_by_view=p90,
        residual_quantiles_px_by_view=quantiles,
        offset_confidence_interval95_px_by_view=intervals,
        accepted_by_view=accepted,
        config=cfg,
    )


__all__ = [
    "BaselineNasalRegistrationSurface",
    "FixedNasalViewRegistration",
    "NASAL_REGISTRATION_VIEWS",
    "NasalViewRegistrationConfig",
    "PROTECTED_REGISTRATION_REGIONS",
    "ViewRegistrationSample",
    "estimate_fixed_nasal_view_offsets",
]

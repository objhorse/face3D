"""Auditable weak appearance evidence for fixed-rig nasal reconstruction."""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Mapping, Sequence

import cv2
import numpy as np


NASAL_SIDE_VIEWS = ("subject-left", "subject-right")
NASAL_TEXTURE_VIEWS = ("front",) + NASAL_SIDE_VIEWS


def _readonly(value: Any, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _finite_positive(name: str, value: float) -> None:
    if not np.isfinite(value) or float(value) <= 0.0:
        raise ValueError(f"{name} must be finite and positive")


def _frozen_numeric_mapping(
    values: Mapping[str, Any],
    *,
    allowed_keys: set[str] | None = None,
) -> Mapping[str, float]:
    result = {}
    for key, value in dict(values).items():
        name = str(key)
        if allowed_keys is not None and name not in allowed_keys:
            raise ValueError(f"unknown mapping key: {name}")
        number = float(value)
        if not np.isfinite(number):
            raise ValueError(f"mapping value must be finite: {name}")
        result[name] = number
    return MappingProxyType(result)


def _deep_freeze(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return _readonly(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _deep_freeze(item) for key, item in dict(value).items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_deep_freeze(item) for item in value)
    if isinstance(value, set):
        return frozenset(_deep_freeze(item) for item in value)
    if isinstance(value, np.generic):
        return _deep_freeze(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    raise ValueError(f"unsupported metadata value: {type(value).__name__}")


def _json_ready(value: Any) -> Any:
    if hasattr(value, "to_dict"):
        return value.to_dict()
    if isinstance(value, np.ndarray):
        return _json_ready(value.tolist())
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (tuple, list, set, frozenset)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.generic):
        return _json_ready(value.item())
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


@dataclass(frozen=True)
class NasalTextureObservationConfig:
    highlight_luma_threshold: int = 245
    shadow_luma_threshold: int = 20
    texture_gradient_floor: float = 0.015
    min_match_confidence: float = 0.5
    reciprocal_tolerance_px: float = 1.5
    max_reprojection_px: float = 2.5
    min_ray_angle_deg: float = 3.0
    min_depth_m: float = 0.12
    max_depth_m: float = 1.50
    epipolar_search_half_length_px: float = 16.0
    patch_radius_px: int = 4
    uniqueness_margin: float = 0.06
    min_pixel_confidence: float = 0.05
    epipolar_step_px: float = 1.0
    loftr_seed_radius_px: float = 8.0
    loftr_score_weight: float = 0.15
    max_model_epipolar_prior_error_px: float = 4.0

    def __post_init__(self) -> None:
        for name in (
            "highlight_luma_threshold",
            "shadow_luma_threshold",
            "patch_radius_px",
        ):
            value = getattr(self, name)
            if isinstance(value, (bool, np.bool_)) or not isinstance(
                value, (int, np.integer)
            ):
                raise ValueError(f"{name} must be an integer")
        if not 1 <= int(self.highlight_luma_threshold) <= 255:
            raise ValueError("highlight_luma_threshold must lie in [1, 255]")
        if not 0 <= int(self.shadow_luma_threshold) < 255:
            raise ValueError("shadow_luma_threshold must lie in [0, 254]")
        if int(self.shadow_luma_threshold) >= int(self.highlight_luma_threshold):
            raise ValueError("shadow threshold must be below highlight threshold")
        for name in (
            "texture_gradient_floor",
            "min_match_confidence",
            "reciprocal_tolerance_px",
            "max_reprojection_px",
            "min_ray_angle_deg",
            "min_depth_m",
            "max_depth_m",
            "epipolar_search_half_length_px",
            "uniqueness_margin",
            "min_pixel_confidence",
            "epipolar_step_px",
            "loftr_seed_radius_px",
            "loftr_score_weight",
            "max_model_epipolar_prior_error_px",
        ):
            _finite_positive(name, float(getattr(self, name)))
        if float(self.min_depth_m) >= float(self.max_depth_m):
            raise ValueError("minimum depth must be below maximum depth")
        if int(self.patch_radius_px) < 2:
            raise ValueError("patch_radius_px must be at least 2")
        if not 0.0 < float(self.min_match_confidence) <= 1.0:
            raise ValueError("min_match_confidence must lie in (0, 1]")
        if float(self.min_ray_angle_deg) > 180.0:
            raise ValueError("min_ray_angle_deg must not exceed 180")
        if float(self.min_pixel_confidence) > 1.0:
            raise ValueError("min_pixel_confidence must not exceed 1")
        if float(self.loftr_score_weight) > 1.0:
            raise ValueError("loftr_score_weight must not exceed 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            name: _json_ready(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class NasalCoordinateProvenance:
    semantic_view: str
    source_size: tuple[int, int]
    work_size: tuple[int, int]
    source_to_work: np.ndarray
    undistorted: bool

    def __post_init__(self) -> None:
        view = str(self.semantic_view)
        if view not in NASAL_TEXTURE_VIEWS:
            raise ValueError(f"unknown semantic_view: {view}")
        source_size = tuple(int(value) for value in self.source_size)
        work_size = tuple(int(value) for value in self.work_size)
        if len(source_size) != 2 or min(source_size) <= 0:
            raise ValueError("source_size must contain two positive values")
        if len(work_size) != 2 or min(work_size) <= 0:
            raise ValueError("work_size must contain two positive values")
        transform = np.asarray(self.source_to_work, dtype=np.float64)
        if transform.shape != (3, 3) or not np.isfinite(transform).all():
            raise ValueError("source_to_work must have finite shape (3, 3)")
        condition = float(np.linalg.cond(transform))
        if not np.isfinite(condition) or condition > 1e8:
            raise ValueError("source_to_work condition number is unsafe")
        homogeneous_scale = float(transform[2, 2])
        relative_scale = max(float(np.max(np.abs(transform))), 1e-300)
        if abs(homogeneous_scale) <= np.finfo(np.float64).eps * relative_scale:
            raise ValueError("source_to_work must use a finite affine scale")
        transform = transform / homogeneous_scale
        if not np.allclose(
            transform[2],
            np.asarray([0.0, 0.0, 1.0]),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("source_to_work must be an affine image transform")
        expected_resize = np.asarray(
            [
                [work_size[0] / float(source_size[0]), 0.0, 0.0],
                [0.0, work_size[1] / float(source_size[1]), 0.0],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float64,
        )
        if not np.allclose(transform, expected_resize, rtol=0.0, atol=1e-12):
            raise ValueError(
                "Release A source_to_work must be the canonical full-frame resize"
            )
        object.__setattr__(self, "semantic_view", view)
        object.__setattr__(self, "source_size", source_size)
        object.__setattr__(self, "work_size", work_size)
        object.__setattr__(self, "source_to_work", _readonly(transform, np.float64))
        if not isinstance(self.undistorted, (bool, np.bool_)):
            raise ValueError("undistorted must be a boolean")
        object.__setattr__(self, "undistorted", bool(self.undistorted))

    def to_dict(self) -> dict[str, Any]:
        return {
            "semantic_view": self.semantic_view,
            "source_size": list(self.source_size),
            "work_size": list(self.work_size),
            "source_to_work": self.source_to_work.tolist(),
            "undistorted": self.undistorted,
        }


def _validate_pixel(
    name: str,
    value: Any,
    provenance: NasalCoordinateProvenance,
) -> np.ndarray:
    point = np.asarray(value, dtype=np.float64)
    if point.shape != (2,) or not np.isfinite(point).all():
        raise ValueError(f"{name} must have finite shape (2,)")
    width, height = provenance.work_size
    if not (0.0 <= point[0] < width and 0.0 <= point[1] < height):
        raise ValueError(f"{name} lies outside its work image")
    return _readonly(point, np.float64)


@dataclass(frozen=True)
class NasalPairMatch:
    side_view: str
    front_pixel: np.ndarray
    side_pixel: np.ndarray
    confidence: float
    semantic_region: str
    source_matcher: str
    provenance_by_view: Mapping[str, NasalCoordinateProvenance]
    diagnostics: Mapping[str, float]

    def __post_init__(self) -> None:
        side = str(self.side_view)
        if side not in NASAL_SIDE_VIEWS:
            raise ValueError(f"side_view must be one of {NASAL_SIDE_VIEWS}")
        provenance = dict(self.provenance_by_view)
        expected = {"front", side}
        if set(provenance) != expected:
            raise ValueError("provenance_by_view must contain front and side_view")
        for view, item in provenance.items():
            if not isinstance(item, NasalCoordinateProvenance):
                raise ValueError("invalid coordinate provenance")
            if item.semantic_view != view:
                raise ValueError("coordinate provenance view mismatch")
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        region = str(self.semantic_region).strip()
        matcher = str(self.source_matcher).strip()
        if not region:
            raise ValueError("semantic_region must not be empty")
        if not matcher:
            raise ValueError("source_matcher must not be empty")
        object.__setattr__(self, "side_view", side)
        object.__setattr__(
            self,
            "front_pixel",
            _validate_pixel("front_pixel", self.front_pixel, provenance["front"]),
        )
        object.__setattr__(
            self,
            "side_pixel",
            _validate_pixel("side_pixel", self.side_pixel, provenance[side]),
        )
        object.__setattr__(self, "confidence", confidence)
        object.__setattr__(self, "semantic_region", region)
        object.__setattr__(self, "source_matcher", matcher)
        object.__setattr__(
            self,
            "provenance_by_view",
            MappingProxyType(provenance),
        )
        object.__setattr__(
            self,
            "diagnostics",
            _frozen_numeric_mapping(self.diagnostics),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "side_view": self.side_view,
            "front_pixel": self.front_pixel.tolist(),
            "side_pixel": self.side_pixel.tolist(),
            "confidence": self.confidence,
            "semantic_region": self.semantic_region,
            "source_matcher": self.source_matcher,
            "provenance_by_view": _json_ready(self.provenance_by_view),
            "diagnostics": dict(self.diagnostics),
        }


@dataclass(frozen=True)
class TrustedNasalObservation:
    pair_match: NasalPairMatch
    point_reference_m: np.ndarray
    covariance_proxy: np.ndarray
    depths_m: Mapping[str, float]
    reprojection_errors_px: Mapping[str, float]
    ray_angle_deg: float
    weight: float
    reference_view: str = "front"
    limits: NasalTextureObservationConfig = field(
        default_factory=NasalTextureObservationConfig
    )

    def __post_init__(self) -> None:
        if not isinstance(self.pair_match, NasalPairMatch):
            raise ValueError("pair_match must be a NasalPairMatch")
        if str(self.reference_view) != "front":
            raise ValueError("reference_view must be front")
        point = np.asarray(self.point_reference_m, dtype=np.float64)
        if point.shape != (3,) or not np.isfinite(point).all():
            raise ValueError("point_reference_m must have finite shape (3,)")
        if float(point[2]) <= 0.0:
            raise ValueError("point_reference_m must have positive depth")
        covariance = np.asarray(self.covariance_proxy, dtype=np.float64)
        if covariance.shape != (3, 3) or not np.isfinite(covariance).all():
            raise ValueError("covariance_proxy must have finite shape (3, 3)")
        if not np.allclose(covariance, covariance.T, rtol=0.0, atol=1e-12):
            raise ValueError("covariance_proxy must be symmetric")
        if float(np.min(np.linalg.eigvalsh(covariance))) < -1e-12:
            raise ValueError("covariance_proxy must be positive semidefinite")
        expected_views = {"front", self.pair_match.side_view}
        if not isinstance(self.limits, NasalTextureObservationConfig):
            raise ValueError("limits must be a NasalTextureObservationConfig")
        if self.pair_match.confidence < float(self.limits.min_match_confidence):
            raise ValueError("pair match confidence is below trusted threshold")
        depths = _frozen_numeric_mapping(
            self.depths_m,
            allowed_keys=expected_views,
        )
        if set(depths) != expected_views or any(
            depth < float(self.limits.min_depth_m)
            or depth > float(self.limits.max_depth_m)
            for depth in depths.values()
        ):
            raise ValueError("all camera observations must have valid positive depth")
        if not np.isclose(
            float(point[2]),
            float(depths["front"]),
            rtol=1e-6,
            atol=1e-8,
        ):
            raise ValueError("point_reference_m Z must match front reference depth")
        reprojection = _frozen_numeric_mapping(
            self.reprojection_errors_px,
            allowed_keys=expected_views,
        )
        if (
            set(reprojection) != expected_views
            or min(reprojection.values()) < 0.0
            or max(reprojection.values()) > float(self.limits.max_reprojection_px)
        ):
            raise ValueError("reprojection_errors_px must contain non-negative pair errors")
        ray_angle = float(self.ray_angle_deg)
        weight = float(self.weight)
        if not (
            float(self.limits.min_ray_angle_deg) <= ray_angle <= 180.0
        ):
            raise ValueError("ray_angle_deg lies outside trusted limits")
        if not np.isfinite(weight) or not 0.0 <= weight <= 1.0:
            raise ValueError("weight must lie in [0, 1]")
        if weight > self.pair_match.confidence + 1e-12:
            raise ValueError("weight cannot exceed pair match confidence")
        object.__setattr__(self, "point_reference_m", _readonly(point, np.float64))
        object.__setattr__(
            self,
            "covariance_proxy",
            _readonly(covariance, np.float64),
        )
        object.__setattr__(self, "depths_m", depths)
        object.__setattr__(self, "reprojection_errors_px", reprojection)
        object.__setattr__(self, "ray_angle_deg", ray_angle)
        object.__setattr__(self, "weight", weight)
        object.__setattr__(self, "reference_view", "front")

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_match": self.pair_match.to_dict(),
            "point_reference_m": self.point_reference_m.tolist(),
            "covariance_proxy": self.covariance_proxy.tolist(),
            "depths_m": dict(self.depths_m),
            "reprojection_errors_px": dict(self.reprojection_errors_px),
            "ray_angle_deg": self.ray_angle_deg,
            "weight": self.weight,
            "reference_view": self.reference_view,
            "limits": self.limits.to_dict(),
        }


@dataclass(frozen=True)
class RejectedNasalObservation:
    pair_match: NasalPairMatch
    reasons: tuple[str, ...]
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.pair_match, NasalPairMatch):
            raise ValueError("pair_match must be a NasalPairMatch")
        reasons = tuple(str(reason).strip() for reason in self.reasons)
        if not reasons or any(not reason for reason in reasons):
            raise ValueError("rejected observation requires non-empty reasons")
        object.__setattr__(self, "reasons", reasons)
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "pair_match": self.pair_match.to_dict(),
            "reasons": list(self.reasons),
            "diagnostics": _json_ready(self.diagnostics),
        }


@dataclass(frozen=True)
class NasalTextureObservationBundle:
    trusted: tuple[TrustedNasalObservation, ...]
    rejected: tuple[RejectedNasalObservation, ...]
    metadata: Mapping[str, Any]

    def __post_init__(self) -> None:
        trusted = tuple(self.trusted)
        if not all(isinstance(value, TrustedNasalObservation) for value in trusted):
            raise ValueError("trusted observations contain an invalid record")
        rejected = tuple(self.rejected)
        if not all(isinstance(value, RejectedNasalObservation) for value in rejected):
            raise ValueError("rejected observations contain an invalid record")
        object.__setattr__(self, "trusted", trusted)
        object.__setattr__(self, "rejected", rejected)
        object.__setattr__(self, "metadata", _deep_freeze(self.metadata))

    @property
    def by_side(self) -> Mapping[str, tuple[TrustedNasalObservation, ...]]:
        return MappingProxyType(
            {
                side: tuple(
                    observation
                    for observation in self.trusted
                    if observation.pair_match.side_view == side
                )
                for side in NASAL_SIDE_VIEWS
            }
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "trusted": [value.to_dict() for value in self.trusted],
            "rejected": [value.to_dict() for value in self.rejected],
            "metadata": _json_ready(self.metadata),
        }


@dataclass(frozen=True)
class NasalTextureConfidenceMaps:
    semantic_support: np.ndarray
    specular_reject: np.ndarray
    shadow_reject: np.ndarray
    texture_strength: np.ndarray
    final_confidence: np.ndarray

    def __post_init__(self) -> None:
        support_raw = np.asarray(self.semantic_support)
        if not np.isfinite(support_raw).all():
            raise ValueError("confidence maps must contain finite values")
        shape = support_raw.shape
        if len(shape) != 2:
            raise ValueError("confidence maps must be two-dimensional")
        for name in (
            "specular_reject",
            "shadow_reject",
            "texture_strength",
            "final_confidence",
        ):
            value = np.asarray(getattr(self, name))
            if value.shape != shape or not np.isfinite(value).all():
                raise ValueError("confidence maps must share one finite shape")
        texture = np.asarray(self.texture_strength, dtype=np.float64)
        confidence = np.asarray(self.final_confidence, dtype=np.float64)
        support = support_raw.astype(bool)
        specular = np.asarray(self.specular_reject, dtype=bool)
        shadow = np.asarray(self.shadow_reject, dtype=bool)
        if np.any((texture < 0.0) | (texture > 1.0)):
            raise ValueError("texture_strength must lie in [0, 1]")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("final_confidence must lie in [0, 1]")
        if np.any(confidence[~support | specular | shadow] != 0.0):
            raise ValueError("final_confidence must be zero outside trusted support")
        if np.any(confidence > texture + 1e-12):
            raise ValueError("final_confidence cannot exceed texture_strength")
        object.__setattr__(
            self,
            "semantic_support",
            _readonly(support, bool),
        )
        object.__setattr__(
            self,
            "specular_reject",
            _readonly(self.specular_reject, bool),
        )
        object.__setattr__(
            self,
            "shadow_reject",
            _readonly(self.shadow_reject, bool),
        )
        object.__setattr__(self, "texture_strength", _readonly(texture, np.float64))
        object.__setattr__(
            self,
            "final_confidence",
            _readonly(confidence, np.float64),
        )

    def to_dict(self) -> dict[str, Any]:
        support = self.semantic_support
        support_count = int(np.count_nonzero(support))
        if support_count:
            texture_mean = float(np.mean(self.texture_strength[support]))
            confidence_mean = float(np.mean(self.final_confidence[support]))
        else:
            texture_mean = 0.0
            confidence_mean = 0.0
        return {
            "shape": list(self.semantic_support.shape),
            "semantic_support_count": support_count,
            "specular_reject_count": int(
                np.count_nonzero(self.specular_reject & support)
            ),
            "shadow_reject_count": int(
                np.count_nonzero(self.shadow_reject & support)
            ),
            "texture_strength_mean": texture_mean,
            "final_confidence_mean": confidence_mean,
        }


def _mask(value: Any, shape: tuple[int, int], name: str) -> np.ndarray:
    array = np.asarray(value)
    if array.shape != shape or not np.isfinite(array).all():
        raise ValueError(f"{name} must have finite image shape")
    return array > 0


def build_nasal_texture_confidence_maps(
    image_rgb: np.ndarray,
    semantic_nose_mask: np.ndarray,
    *,
    face_mask: np.ndarray | None = None,
    nostril_mask: np.ndarray | None = None,
    projected_support: np.ndarray,
    config: NasalTextureObservationConfig | None = None,
) -> NasalTextureConfidenceMaps:
    """Separate usable nasal skin from lighting and texture failure modes."""
    cfg = config or NasalTextureObservationConfig()
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("image_rgb must have finite shape (H, W, 3)")
    if image.dtype != np.uint8:
        raise ValueError("image_rgb must use uint8 RGB values")
    shape = image.shape[:2]
    semantic = _mask(semantic_nose_mask, shape, "semantic_nose_mask")
    if face_mask is not None:
        semantic &= _mask(face_mask, shape, "face_mask")
    semantic &= _mask(projected_support, shape, "projected_support")

    gray_u8 = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    gray = gray_u8.astype(np.float64) / 255.0
    specular = gray_u8 >= int(cfg.highlight_luma_threshold)
    supported_non_specular_luma = gray_u8[semantic & ~specular]
    bright_reference_luma = (
        float(np.percentile(supported_non_specular_luma, 98.0))
        if supported_non_specular_luma.size
        else float(cfg.shadow_luma_threshold)
    )
    adaptive_shadow_threshold = max(
        float(cfg.shadow_luma_threshold),
        0.45 * bright_reference_luma,
    )
    shadow = gray_u8.astype(np.float64) <= adaptive_shadow_threshold
    if nostril_mask is not None:
        shadow |= _mask(nostril_mask, shape, "nostril_mask")
    reject_radius = int(cfg.patch_radius_px)
    reject_kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * reject_radius + 1, 2 * reject_radius + 1),
    )
    specular = cv2.dilate(specular.astype(np.uint8), reject_kernel) > 0
    shadow = cv2.dilate(shadow.astype(np.uint8), reject_kernel) > 0

    smoothed = cv2.GaussianBlur(gray, (0, 0), sigmaX=0.8, sigmaY=0.8)
    grad_x = cv2.Sobel(smoothed, cv2.CV_64F, 1, 0, ksize=3) / 8.0
    grad_y = cv2.Sobel(smoothed, cv2.CV_64F, 0, 1, ksize=3) / 8.0
    magnitude = np.hypot(grad_x, grad_y)
    floor = float(cfg.texture_gradient_floor)
    patch_sigma = max(1.0, 0.5 * int(cfg.patch_radius_px))
    local_energy = np.sqrt(
        np.maximum(
            cv2.GaussianBlur(
                magnitude * magnitude,
                (0, 0),
                sigmaX=patch_sigma,
                sigmaY=patch_sigma,
            ),
            0.0,
        )
    )
    texture = np.clip(
        (local_energy - 0.25 * floor) / (2.0 * floor),
        0.0,
        1.0,
    )
    texture = np.clip(texture, 0.0, 1.0)

    confidence = texture * semantic.astype(np.float64)
    confidence[specular | shadow] = 0.0
    return NasalTextureConfidenceMaps(
        semantic_support=semantic,
        specular_reject=specular,
        shadow_reject=shadow,
        texture_strength=texture,
        final_confidence=confidence,
    )


@dataclass(frozen=True)
class NasalEpipolarSeed:
    front_pixel: np.ndarray
    predicted_side_pixel: np.ndarray
    semantic_region: str
    baseline_vertex_index: int
    side_from_front_jacobian: np.ndarray = field(
        default_factory=lambda: np.eye(2, dtype=np.float64)
    )

    def __post_init__(self) -> None:
        region = str(self.semantic_region).strip()
        if not region:
            raise ValueError("semantic_region must not be empty")
        if isinstance(self.baseline_vertex_index, bool) or not isinstance(
            self.baseline_vertex_index, (int, np.integer)
        ):
            raise ValueError("baseline_vertex_index must be an integer")
        if int(self.baseline_vertex_index) < 0:
            raise ValueError("baseline_vertex_index must be non-negative")
        front = np.asarray(self.front_pixel, dtype=np.float64)
        side = np.asarray(self.predicted_side_pixel, dtype=np.float64)
        if front.shape != (2,) or not np.isfinite(front).all():
            raise ValueError("front_pixel must have finite shape (2,)")
        if side.shape != (2,) or not np.isfinite(side).all():
            raise ValueError("predicted_side_pixel must have finite shape (2,)")
        jacobian = np.asarray(self.side_from_front_jacobian, dtype=np.float64)
        if jacobian.shape != (2, 2) or not np.isfinite(jacobian).all():
            raise ValueError("side_from_front_jacobian must have finite shape (2, 2)")
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        if float(singular_values[-1]) < 0.15 or float(singular_values[0]) > 6.0:
            raise ValueError("side_from_front_jacobian is outside safe local scale limits")
        object.__setattr__(self, "front_pixel", _readonly(front, np.float64))
        object.__setattr__(
            self,
            "predicted_side_pixel",
            _readonly(side, np.float64),
        )
        object.__setattr__(self, "semantic_region", region)
        object.__setattr__(
            self,
            "baseline_vertex_index",
            int(self.baseline_vertex_index),
        )
        object.__setattr__(
            self,
            "side_from_front_jacobian",
            _readonly(jacobian, np.float64),
        )


@dataclass(frozen=True)
class RejectedNasalMatch:
    seed: NasalEpipolarSeed
    reason: str
    diagnostics: Mapping[str, Any]

    def __post_init__(self) -> None:
        if not isinstance(self.seed, NasalEpipolarSeed):
            raise ValueError("seed must be a NasalEpipolarSeed")
        reason = str(self.reason).strip()
        if not reason:
            raise ValueError("rejection reason must not be empty")
        object.__setattr__(self, "reason", reason)
        object.__setattr__(self, "diagnostics", _deep_freeze(self.diagnostics))

    def to_dict(self) -> dict[str, Any]:
        return {
            "seed": {
                "front_pixel": self.seed.front_pixel.tolist(),
                "predicted_side_pixel": self.seed.predicted_side_pixel.tolist(),
                "semantic_region": self.seed.semantic_region,
                "baseline_vertex_index": self.seed.baseline_vertex_index,
                "side_from_front_jacobian": self.seed.side_from_front_jacobian.tolist(),
            },
            "reason": self.reason,
            "diagnostics": _json_ready(self.diagnostics),
        }


@dataclass(frozen=True)
class NasalEpipolarMatchResult:
    matches: tuple[NasalPairMatch, ...]
    rejected: tuple[RejectedNasalMatch, ...]

    def __post_init__(self) -> None:
        matches = tuple(self.matches)
        rejected = tuple(self.rejected)
        if not all(isinstance(value, NasalPairMatch) for value in matches):
            raise ValueError("matches contain an invalid record")
        if not all(isinstance(value, RejectedNasalMatch) for value in rejected):
            raise ValueError("rejected matches contain an invalid record")
        object.__setattr__(self, "matches", matches)
        object.__setattr__(self, "rejected", rejected)

    def to_dict(self) -> dict[str, Any]:
        return {
            "matches": [value.to_dict() for value in self.matches],
            "rejected": [value.to_dict() for value in self.rejected],
        }


def _gray_float(image_rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(image_rgb)
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("matching images must use uint8 RGB shape (H, W, 3)")
    return cv2.cvtColor(image, cv2.COLOR_RGB2GRAY).astype(np.float64) / 255.0


def _validate_work_image_size(
    image_rgb: np.ndarray,
    provenance: NasalCoordinateProvenance,
    label: str,
) -> None:
    actual = (int(image_rgb.shape[1]), int(image_rgb.shape[0]))
    if tuple(provenance.work_size) != actual:
        raise ValueError(
            f"{label} image size {actual} disagrees with coordinate provenance "
            f"{provenance.work_size}"
        )


def _patch_descriptor(
    gray: np.ndarray,
    point: np.ndarray,
    radius: int,
    sampling_jacobian: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    center_point = np.asarray(point, dtype=np.float64)
    jacobian = (
        np.eye(2, dtype=np.float64)
        if sampling_jacobian is None
        else np.asarray(sampling_jacobian, dtype=np.float64).reshape(2, 2)
    )
    offsets = np.arange(-radius, radius + 1, dtype=np.float64)
    grid_x, grid_y = np.meshgrid(offsets, offsets)
    canonical = np.column_stack((grid_x.reshape(-1), grid_y.reshape(-1)))
    sampled = center_point[None, :] + canonical @ jacobian.T
    if (
        np.min(sampled[:, 0]) < 1.0
        or np.min(sampled[:, 1]) < 1.0
        or np.max(sampled[:, 0]) >= gray.shape[1] - 1.0
        or np.max(sampled[:, 1]) >= gray.shape[0] - 1.0
    ):
        return None
    size = 2 * radius + 1
    map_x = sampled[:, 0].reshape(size, size).astype(np.float32)
    map_y = sampled[:, 1].reshape(size, size).astype(np.float32)
    patch = cv2.remap(
        gray.astype(np.float32),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT101,
    ).astype(np.float64)
    grad_x = cv2.Sobel(patch, cv2.CV_64F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(patch, cv2.CV_64F, 0, 1, ksize=3)
    gradient = np.concatenate((grad_x.reshape(-1), grad_y.reshape(-1)))
    gradient -= float(np.mean(gradient))
    norm = float(np.linalg.norm(gradient))
    if norm <= 1e-8:
        return None
    gradient /= norm
    center = float(patch[radius, radius])
    census = (patch.reshape(-1) >= center)
    census = np.delete(census, radius * size + radius)
    return gradient, census


def _descriptor_similarity(
    first: tuple[np.ndarray, np.ndarray],
    second: tuple[np.ndarray, np.ndarray],
) -> float:
    gradient_score = 0.5 * (float(np.dot(first[0], second[0])) + 1.0)
    census_score = float(np.mean(first[1] == second[1]))
    return float(np.clip(0.65 * gradient_score + 0.35 * census_score, 0.0, 1.0))


def _confidence_at(maps: NasalTextureConfidenceMaps, point: np.ndarray) -> float:
    x, y = np.rint(point).astype(np.int64)
    if x < 0 or y < 0 or x >= maps.final_confidence.shape[1] or y >= maps.final_confidence.shape[0]:
        return 0.0
    return float(maps.final_confidence[y, x])


def _epipolar_candidates(
    line: np.ndarray,
    predicted: np.ndarray,
    half_length: float,
    step: float,
) -> tuple[np.ndarray, np.ndarray]:
    a, b, c = np.asarray(line, dtype=np.float64).reshape(3)
    denominator = float(a * a + b * b)
    if denominator <= 1e-12:
        return np.empty((0, 2), dtype=np.float64), np.empty(0, dtype=np.float64)
    distance = float((a * predicted[0] + b * predicted[1] + c) / denominator)
    anchor = predicted - np.asarray([a, b]) * distance
    direction = np.asarray([-b, a], dtype=np.float64) / np.sqrt(denominator)
    offsets = np.arange(-half_length, half_length + 0.5 * step, step)
    points = anchor[None, :] + offsets[:, None] * direction[None, :]
    inside = np.linalg.norm(points - predicted[None, :], axis=1) <= half_length + 1e-9
    return points[inside], offsets[inside]


def _work_intrinsics_from_provenance(
    camera: Any,
    provenance: NasalCoordinateProvenance,
) -> np.ndarray:
    if not provenance.undistorted:
        raise ValueError("nasal matching and triangulation require undistorted pixels")
    if tuple(camera.image_size) != tuple(provenance.source_size):
        raise ValueError("coordinate provenance source size must match calibrated camera")
    intrinsics = provenance.source_to_work @ np.asarray(camera.K, dtype=np.float64)
    if (
        intrinsics.shape != (3, 3)
        or not np.isfinite(intrinsics).all()
        or abs(float(np.linalg.det(intrinsics))) <= 1e-12
    ):
        raise ValueError("coordinate provenance produces invalid work intrinsics")
    return intrinsics


def _fixed_rig_fundamental(
    rig: Any,
    side_view: str,
    provenance_by_view: Mapping[str, NasalCoordinateProvenance],
    rig_view_by_semantic: Mapping[str, str],
) -> np.ndarray:
    from src.cross_view_geometry import relative_camera_transform
    from src.geometry.profile_triangulation import ProfileRig

    if not isinstance(rig, ProfileRig):
        raise ValueError("rig must be a validated ProfileRig")
    _validate_rig_view_mapping(rig, rig_view_by_semantic)
    front_camera = rig.cameras_by_view[rig_view_by_semantic["front"]]
    side_camera = rig.cameras_by_view[rig_view_by_semantic[side_view]]
    front_k = _work_intrinsics_from_provenance(
        front_camera,
        provenance_by_view["front"],
    )
    side_k = _work_intrinsics_from_provenance(
        side_camera,
        provenance_by_view[side_view],
    )
    rotation, translation = relative_camera_transform(front_camera, side_camera)
    tx, ty, tz = np.asarray(translation, dtype=np.float64).reshape(3)
    skew = np.asarray(
        [[0.0, -tz, ty], [tz, 0.0, -tx], [-ty, tx, 0.0]],
        dtype=np.float64,
    )
    fundamental = np.linalg.inv(side_k).T @ skew @ rotation @ np.linalg.inv(front_k)
    norm = float(np.linalg.norm(fundamental))
    if not np.isfinite(norm) or norm <= 1e-12:
        raise ValueError("fixed rig produces a degenerate fundamental matrix")
    return fundamental / norm


def fixed_rig_fundamental(
    rig: Any,
    side_view: str,
    provenance_by_view: Mapping[str, NasalCoordinateProvenance],
    rig_view_by_semantic: Mapping[str, str],
) -> np.ndarray:
    """Return the fixed-rig fundamental matrix in work-image coordinates."""
    return _readonly(
        _fixed_rig_fundamental(
            rig,
            side_view,
            provenance_by_view,
            rig_view_by_semantic,
        ),
        np.float64,
    )


def _validate_rig_view_mapping(
    rig: Any,
    rig_view_by_semantic: Mapping[str, str],
) -> None:
    expected_camera_view = {
        "front": "front",
        "subject-left": "left",
        "subject-right": "right",
    }
    mapping = dict(rig_view_by_semantic)
    if set(mapping) != set(NASAL_TEXTURE_VIEWS):
        raise ValueError("rig view mapping must contain every semantic view")
    if len(set(mapping.values())) != len(NASAL_TEXTURE_VIEWS):
        raise ValueError("rig view mapping must be one-to-one")
    for semantic, rig_view in mapping.items():
        if rig_view not in rig.cameras_by_view:
            raise ValueError("rig view mapping refers to a missing camera")
        if str(rig.cameras_by_view[rig_view].view) != expected_camera_view[semantic]:
            raise ValueError("rig camera view semantics disagree with semantic mapping")


def _loftr_support(
    front_pixel: np.ndarray,
    side_pixel: np.ndarray,
    matches: Sequence[NasalPairMatch],
    side_view: str,
    radius: float,
) -> float:
    support = 0.0
    scale = max(float(radius), 1e-6)
    for match in matches:
        if match.side_view != side_view:
            continue
        front_distance = float(np.linalg.norm(match.front_pixel - front_pixel))
        side_distance = float(np.linalg.norm(match.side_pixel - side_pixel))
        if front_distance > radius or side_distance > radius:
            continue
        support = max(
            support,
            float(match.confidence)
            * float(np.exp(-0.5 * (front_distance / scale) ** 2))
            * float(np.exp(-0.5 * (side_distance / scale) ** 2)),
        )
    return support


def _search_epipolar_patch(
    source_descriptor: tuple[np.ndarray, np.ndarray],
    target_gray: np.ndarray,
    target_confidence: NasalTextureConfidenceMaps,
    line: np.ndarray,
    predicted: np.ndarray,
    config: NasalTextureObservationConfig,
    *,
    front_pixel: np.ndarray,
    side_view: str,
    loftr_seed_matches: Sequence[NasalPairMatch],
    target_sampling_jacobian: np.ndarray | None = None,
) -> dict[str, Any]:
    points, offsets = _epipolar_candidates(
        line,
        predicted,
        float(config.epipolar_search_half_length_px),
        float(config.epipolar_step_px),
    )
    candidates: list[tuple[float, float, np.ndarray, float, float]] = []
    for point, offset in zip(points, offsets):
        pixel_confidence = _confidence_at(target_confidence, point)
        if pixel_confidence < float(config.min_pixel_confidence):
            continue
        descriptor = _patch_descriptor(
            target_gray,
            point,
            int(config.patch_radius_px),
            target_sampling_jacobian,
        )
        if descriptor is None:
            continue
        base_score = _descriptor_similarity(source_descriptor, descriptor)
        loftr = _loftr_support(
            front_pixel,
            point,
            loftr_seed_matches,
            side_view,
            float(config.loftr_seed_radius_px),
        )
        blend = float(config.loftr_score_weight)
        score = (1.0 - blend) * base_score + blend * loftr
        candidates.append((score, abs(float(offset)), point, pixel_confidence, loftr))
    if not candidates:
        return {"passed": False, "reason": "no_valid_epipolar_candidate"}
    candidates.sort(key=lambda item: (-item[0], item[1], item[2][0], item[2][1]))
    best = candidates[0]
    exclusion = max(1.0, 0.5 * int(config.patch_radius_px))
    independent = [
        item for item in candidates[1:]
        if np.linalg.norm(item[2] - best[2]) > exclusion
    ]
    second_score = independent[0][0] if independent else 0.0
    margin = float(best[0] - second_score)
    return {
        "passed": margin >= float(config.uniqueness_margin),
        "reason": "accepted" if margin >= float(config.uniqueness_margin) else "no_unique_match",
        "point": best[2],
        "score": float(best[0]),
        "second_score": float(second_score),
        "uniqueness_margin": margin,
        "pixel_confidence": float(best[3]),
        "loftr_support": float(best[4]),
        "search_displacement_px": float(np.linalg.norm(best[2] - predicted)),
        "candidate_count": len(candidates),
    }


def match_model_guided_nasal_pair(
    front_image_rgb: np.ndarray,
    side_image_rgb: np.ndarray,
    *,
    side_view: str,
    rig: Any,
    seeds: Sequence[NasalEpipolarSeed],
    front_confidence: NasalTextureConfidenceMaps,
    side_confidence: NasalTextureConfidenceMaps,
    provenance_by_view: Mapping[str, NasalCoordinateProvenance],
    loftr_seed_matches: Sequence[NasalPairMatch] = (),
    rig_view_by_semantic: Mapping[str, str] | None = None,
    config: NasalTextureObservationConfig | None = None,
) -> NasalEpipolarMatchResult:
    """Match fixed baseline samples along short rig-defined epipolar segments."""
    cfg = config or NasalTextureObservationConfig()
    if side_view not in NASAL_SIDE_VIEWS:
        raise ValueError(f"side_view must be one of {NASAL_SIDE_VIEWS}")
    provenance = dict(provenance_by_view)
    if set(provenance) != {"front", side_view}:
        raise ValueError("provenance must contain front and side view")
    view_map = dict(
        rig_view_by_semantic
        or {"front": "front", "subject-left": "left", "subject-right": "right"}
    )
    _validate_rig_view_mapping(rig, view_map)
    if set(view_map) != set(NASAL_TEXTURE_VIEWS):
        raise ValueError("rig view mapping must contain every semantic view")
    fundamental = _fixed_rig_fundamental(
        rig,
        side_view,
        provenance,
        view_map,
    )
    front_gray = _gray_float(front_image_rgb)
    side_gray = _gray_float(side_image_rgb)
    _validate_work_image_size(front_image_rgb, provenance["front"], "front")
    _validate_work_image_size(side_image_rgb, provenance[side_view], side_view)
    if front_confidence.final_confidence.shape != front_gray.shape:
        raise ValueError("front confidence map shape does not match image")
    if side_confidence.final_confidence.shape != side_gray.shape:
        raise ValueError("side confidence map shape does not match image")

    accepted_candidates: list[tuple[NasalEpipolarSeed, NasalPairMatch]] = []
    rejected: list[RejectedNasalMatch] = []
    for seed in tuple(seeds):
        if not isinstance(seed, NasalEpipolarSeed):
            raise ValueError("seeds contain an invalid record")
        source_confidence = _confidence_at(front_confidence, seed.front_pixel)
        descriptor = _patch_descriptor(
            front_gray,
            seed.front_pixel,
            int(cfg.patch_radius_px),
        )
        if source_confidence < float(cfg.min_pixel_confidence):
            rejected.append(RejectedNasalMatch(seed, "source_pixel_untrusted", {}))
            continue
        if descriptor is None:
            rejected.append(RejectedNasalMatch(seed, "source_patch_untextured", {}))
            continue
        line = fundamental @ np.append(seed.front_pixel, 1.0)
        line_norm = max(float(np.linalg.norm(line[:2])), 1e-12)
        prior_epipolar_error = abs(
            float(np.dot(line, np.append(seed.predicted_side_pixel, 1.0)))
        ) / line_norm
        if prior_epipolar_error > min(
            float(cfg.max_model_epipolar_prior_error_px),
            float(cfg.epipolar_search_half_length_px),
        ):
            rejected.append(
                RejectedNasalMatch(
                    seed,
                    "model_epipolar_prior_inconsistent",
                    {"model_epipolar_prior_error_px": prior_epipolar_error},
                )
            )
            continue
        forward = _search_epipolar_patch(
            descriptor,
            side_gray,
            side_confidence,
            line,
            seed.predicted_side_pixel,
            cfg,
            front_pixel=seed.front_pixel,
            side_view=side_view,
            loftr_seed_matches=loftr_seed_matches,
            target_sampling_jacobian=seed.side_from_front_jacobian,
        )
        if not forward["passed"]:
            rejected.append(
                RejectedNasalMatch(seed, str(forward["reason"]), forward)
            )
            continue
        side_pixel = np.asarray(forward["point"], dtype=np.float64)
        reverse_descriptor = _patch_descriptor(
            side_gray,
            side_pixel,
            int(cfg.patch_radius_px),
            seed.side_from_front_jacobian,
        )
        reverse_line = fundamental.T @ np.append(side_pixel, 1.0)
        reverse = _search_epipolar_patch(
            reverse_descriptor,
            front_gray,
            front_confidence,
            reverse_line,
            seed.front_pixel,
            cfg,
            front_pixel=side_pixel,
            side_view=side_view,
            loftr_seed_matches=(),
            target_sampling_jacobian=None,
        ) if reverse_descriptor is not None else {"passed": False}
        reciprocal_error = (
            float(np.linalg.norm(np.asarray(reverse["point"]) - seed.front_pixel))
            if reverse.get("passed") and "point" in reverse
            else float("inf")
        )
        if reciprocal_error > float(cfg.reciprocal_tolerance_px):
            rejected.append(
                RejectedNasalMatch(
                    seed,
                    "reciprocal_match_failed",
                    {**forward, "reciprocal_error_px": reciprocal_error},
                )
            )
            continue
        target_confidence = float(forward["pixel_confidence"])
        evidence_factor = 0.5 + 0.5 * np.sqrt(
            max(0.0, source_confidence * target_confidence)
        )
        confidence = float(np.clip(float(forward["score"]) * evidence_factor, 0.0, 1.0))
        if confidence < float(cfg.min_match_confidence):
            rejected.append(
                RejectedNasalMatch(
                    seed,
                    "match_confidence_low",
                    {**forward, "match_confidence": confidence},
                )
            )
            continue
        epipolar_error = abs(float(np.dot(line, np.append(side_pixel, 1.0)))) / line_norm
        diagnostics = {
            key: value
            for key, value in forward.items()
            if key not in {"point", "passed", "reason"}
        }
        diagnostics.update(
            {
                "reciprocal_error_px": reciprocal_error,
                "epipolar_error_px": epipolar_error,
                "baseline_vertex_index": float(seed.baseline_vertex_index),
                "model_epipolar_prior_error_px": prior_epipolar_error,
            }
        )
        accepted_candidates.append(
            (
                seed,
                NasalPairMatch(
                side_view=side_view,
                front_pixel=seed.front_pixel,
                side_pixel=side_pixel,
                confidence=confidence,
                semantic_region=seed.semantic_region,
                source_matcher=(
                    "model_guided_gradient_census_loftr"
                    if loftr_seed_matches
                    else "model_guided_gradient_census"
                ),
                provenance_by_view=provenance,
                diagnostics=diagnostics,
                ),
            )
        )
    matches: list[NasalPairMatch] = []
    used_vertices: set[int] = set()
    used_front_pixels: list[np.ndarray] = []
    used_side_pixels: list[np.ndarray] = []
    for seed, match in sorted(
        accepted_candidates,
        key=lambda item: (-item[1].confidence, item[0].baseline_vertex_index),
    ):
        if seed.baseline_vertex_index in used_vertices:
            rejected.append(
                RejectedNasalMatch(seed, "duplicate_baseline_anchor", match.diagnostics)
            )
            continue
        if any(
            np.linalg.norm(match.front_pixel - pixel)
            <= float(cfg.reciprocal_tolerance_px)
            for pixel in used_front_pixels
        ):
            rejected.append(
                RejectedNasalMatch(seed, "front_pixel_not_one_to_one", match.diagnostics)
            )
            continue
        if any(
            np.linalg.norm(match.side_pixel - pixel)
            <= float(cfg.reciprocal_tolerance_px)
            for pixel in used_side_pixels
        ):
            rejected.append(
                RejectedNasalMatch(seed, "side_pixel_not_one_to_one", match.diagnostics)
            )
            continue
        used_vertices.add(seed.baseline_vertex_index)
        used_front_pixels.append(match.front_pixel)
        used_side_pixels.append(match.side_pixel)
        matches.append(match)
    return NasalEpipolarMatchResult(tuple(matches), tuple(rejected))


def triangulate_nasal_pair_matches(
    matches: Sequence[NasalPairMatch],
    rig: Any,
    *,
    observed_work_size_by_semantic_view: Mapping[str, tuple[int, int]],
    rig_view_by_semantic: Mapping[str, str] | None = None,
    intrinsics_by_semantic_view: Mapping[str, np.ndarray] | None = None,
    config: NasalTextureObservationConfig | None = None,
) -> NasalTextureObservationBundle:
    """Convert accepted image pairs into audited metric observations."""
    from src.geometry.profile_triangulation import (
        ProfileRig,
        TriangulationThresholds,
        triangulate_profile_point,
    )

    if not isinstance(rig, ProfileRig):
        raise ValueError("rig must be a validated ProfileRig")
    cfg = config or NasalTextureObservationConfig()
    view_map = dict(
        rig_view_by_semantic
        or {"front": "front", "subject-left": "left", "subject-right": "right"}
    )
    _validate_rig_view_mapping(rig, view_map)
    observed_sizes = {
        str(view): tuple(int(value) for value in size)
        for view, size in dict(observed_work_size_by_semantic_view).items()
    }
    if set(observed_sizes) != set(NASAL_TEXTURE_VIEWS):
        raise ValueError("observed work sizes must contain every semantic view")
    if any(len(size) != 2 or min(size) <= 0 for size in observed_sizes.values()):
        raise ValueError("observed work sizes must contain positive width and height")
    trusted: list[TrustedNasalObservation] = []
    rejected: list[RejectedNasalObservation] = []
    for pair in tuple(matches):
        if not isinstance(pair, NasalPairMatch):
            raise ValueError("matches contain an invalid record")
        semantic_views = ("front", pair.side_view)
        for semantic_view in semantic_views:
            if pair.provenance_by_view[semantic_view].work_size != observed_sizes[semantic_view]:
                raise ValueError(
                    "coordinate provenance work size disagrees with observed image size"
                )
        rig_views = {view: view_map[view] for view in semantic_views}
        intrinsics: dict[str, np.ndarray] = {}
        for semantic_view, rig_view in rig_views.items():
            derived = _work_intrinsics_from_provenance(
                rig.cameras_by_view[rig_view],
                pair.provenance_by_view[semantic_view],
            )
            if intrinsics_by_semantic_view is None:
                intrinsics[rig_view] = derived
                continue
            supplied = np.asarray(
                intrinsics_by_semantic_view[semantic_view],
                dtype=np.float64,
            )
            if supplied.shape != (3, 3) or not np.allclose(
                supplied,
                derived,
                rtol=1e-9,
                atol=1e-9,
            ):
                raise ValueError("supplied work intrinsics disagree with coordinate provenance")
            intrinsics[rig_view] = supplied
        observations = {
            rig_views["front"]: pair.front_pixel,
            rig_views[pair.side_view]: pair.side_pixel,
        }
        report = triangulate_profile_point(
            observations,
            rig,
            intrinsics_by_view=intrinsics,
            thresholds=TriangulationThresholds(
                max_reprojection_px=float(cfg.max_reprojection_px),
                min_ray_angle_deg=float(cfg.min_ray_angle_deg),
                min_depth_m=float(cfg.min_depth_m),
                max_depth_m=float(cfg.max_depth_m),
            ),
        )
        if not report["passed"]:
            rejected.append(
                RejectedNasalObservation(
                    pair_match=pair,
                    reasons=tuple(report["issues"]),
                    diagnostics=report,
                )
            )
            continue
        point = np.asarray(report["point_reference_m"], dtype=np.float64)
        reprojection = {
            semantic_view: float(
                report["reprojection_errors_px"][rig_views[semantic_view]]
            )
            for semantic_view in semantic_views
        }
        depths = {
            semantic_view: float(report["depths_m"][rig_views[semantic_view]])
            for semantic_view in semantic_views
        }
        angle = float(report["min_ray_angle_deg"])
        focal = float(
            np.mean([intrinsics[view][0, 0] for view in observations])
        )
        sigma = (
            max(float(report["reprojection_p90_px"]), 0.25)
            * float(point[2])
            / max(focal * np.sin(np.deg2rad(angle)), 1e-8)
        )
        weight = float(
            pair.confidence
            * np.exp(
                -float(report["reprojection_p90_px"])
                / max(float(cfg.max_reprojection_px), 1e-8)
            )
        )
        trusted.append(
            TrustedNasalObservation(
                pair_match=pair,
                point_reference_m=point,
                covariance_proxy=np.eye(3, dtype=np.float64) * sigma * sigma,
                depths_m=depths,
                reprojection_errors_px=reprojection,
                ray_angle_deg=angle,
                weight=min(weight, pair.confidence),
                limits=cfg,
            )
        )
    return NasalTextureObservationBundle(
        trusted=tuple(trusted),
        rejected=tuple(rejected),
        metadata={
            "input_matches": len(tuple(matches)),
            "trusted_matches": len(trusted),
            "rejected_matches": len(rejected),
            "rig_view_by_semantic": view_map,
        },
    )


__all__ = [
    "NASAL_SIDE_VIEWS",
    "NASAL_TEXTURE_VIEWS",
    "NasalCoordinateProvenance",
    "NasalEpipolarMatchResult",
    "NasalEpipolarSeed",
    "NasalPairMatch",
    "RejectedNasalObservation",
    "RejectedNasalMatch",
    "NasalTextureConfidenceMaps",
    "NasalTextureObservationBundle",
    "NasalTextureObservationConfig",
    "TrustedNasalObservation",
    "build_nasal_texture_confidence_maps",
    "fixed_rig_fundamental",
    "match_model_guided_nasal_pair",
    "triangulate_nasal_pair_matches",
]

"""Outer-alar curve targets derived from the existing three-view audit."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np

from src.geometry.nasal_observations import (
    NASAL_VIEWS,
    NasalObservationBundle,
)


ALAR_TARGET_NAMES = (
    "front_subject_left_alar",
    "front_subject_right_alar",
    "subject_left_alar_profile",
    "subject_right_alar_profile",
)


def _readonly(value: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


def _resample_curve(curve: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(curve, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) < 2:
        raise ValueError("alar curve must have finite shape (N, 2), N >= 2")
    if not np.isfinite(points).all():
        raise ValueError("alar curve contains non-finite coordinates")
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    usable = lengths > 1e-8
    if not np.any(usable):
        raise ValueError("alar curve has zero arclength")
    starts = points[:-1][usable]
    ends = points[1:][usable]
    lengths = lengths[usable]
    cumulative = np.cumsum(lengths)
    targets = np.linspace(0.0, float(cumulative[-1]), int(count))
    indices = np.searchsorted(cumulative, targets, side="right")
    indices = np.minimum(indices, len(lengths) - 1)
    previous = np.r_[0.0, cumulative[:-1]]
    alpha = (targets - previous[indices]) / lengths[indices]
    return (
        (1.0 - alpha[:, None]) * starts[indices]
        + alpha[:, None] * ends[indices]
    )


def _profile_alar_segment(
    curve: np.ndarray,
    alar_anchor: np.ndarray,
    fraction: float,
) -> np.ndarray:
    points = np.asarray(curve, dtype=np.float64)
    anchor = np.asarray(alar_anchor, dtype=np.float64).reshape(2)
    if np.linalg.norm(points[0] - anchor) < np.linalg.norm(points[-1] - anchor):
        points = points[::-1]
    lengths = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(lengths)]
    threshold = (1.0 - float(fraction)) * float(cumulative[-1])
    start = max(int(np.searchsorted(cumulative, threshold)) - 1, 0)
    segment = points[start:]
    if len(segment) < 4:
        raise ValueError("profile curve has insufficient outer-alar support")
    return segment


def _lower_alar_segment(curve: np.ndarray, fraction: float) -> np.ndarray:
    points = np.asarray(curve, dtype=np.float64)
    y_threshold = float(np.quantile(points[:, 1], 1.0 - float(fraction)))
    segment = points[points[:, 1] >= y_threshold]
    if len(segment) < 4:
        raise ValueError("front curve has insufficient lower-alar support")
    return segment


def _rasterized_distance(
    curve: np.ndarray,
    shape: tuple[int, int],
    outward_sign: float,
) -> np.ndarray:
    height, width = (int(shape[0]), int(shape[1]))
    raster = np.zeros((height, width), dtype=np.uint8)
    rounded = np.rint(curve).astype(np.int32)
    rounded[:, 0] = np.clip(rounded[:, 0], 0, width - 1)
    rounded[:, 1] = np.clip(rounded[:, 1], 0, height - 1)
    cv2.polylines(
        raster,
        [rounded.reshape(-1, 1, 2)],
        isClosed=False,
        color=255,
        thickness=1,
        lineType=cv2.LINE_8,
    )
    unsigned = cv2.distanceTransform(
        np.where(raster > 0, 0, 255).astype(np.uint8),
        cv2.DIST_L2,
        cv2.DIST_MASK_PRECISE,
    ).astype(np.float64)

    # The sign is defined by the horizontal side of the ordered visible curve.
    # Magnitude remains the Euclidean distance, so steep lower-rim sections are
    # not treated as horizontal landmark offsets.
    rows = np.rint(curve[:, 1]).astype(np.int32)
    row_x = {}
    for row in np.unique(rows):
        row_x[int(row)] = float(np.median(curve[rows == row, 0]))
    known_y = np.asarray(sorted(row_x), dtype=np.float64)
    known_x = np.asarray([row_x[int(row)] for row in known_y], dtype=np.float64)
    all_y = np.arange(height, dtype=np.float64)
    x_at_y = np.interp(all_y, known_y, known_x)
    x_grid = np.arange(width, dtype=np.float64)[None, :]
    signed_side = float(outward_sign) * (x_grid - x_at_y[:, None])
    sign = np.where(signed_side >= 0.0, 1.0, -1.0)
    return unsigned * sign


@dataclass(frozen=True)
class AlarCurveTarget:
    name: str
    semantic_view: str
    boundary_name: str
    source_labels: tuple[str, ...]
    curve_work: np.ndarray
    signed_distance_field: np.ndarray
    confidence_field: np.ndarray

    def __post_init__(self) -> None:
        if self.name not in ALAR_TARGET_NAMES:
            raise ValueError(f"unknown alar target: {self.name}")
        if self.semantic_view not in NASAL_VIEWS:
            raise ValueError(f"unknown alar semantic view: {self.semantic_view}")
        curve = np.asarray(self.curve_work, dtype=np.float64)
        distance = np.asarray(self.signed_distance_field, dtype=np.float64)
        confidence = np.asarray(self.confidence_field, dtype=np.float64)
        if curve.ndim != 2 or curve.shape[1] != 2 or len(curve) < 4:
            raise ValueError("alar target curve must have shape (N, 2), N >= 4")
        if distance.ndim != 2 or confidence.shape != distance.shape:
            raise ValueError("alar distance and confidence fields must share shape")
        if not all(np.isfinite(value).all() for value in (curve, distance, confidence)):
            raise ValueError("alar target contains non-finite values")
        if np.any((confidence < 0.0) | (confidence > 1.0)):
            raise ValueError("alar confidence must lie in [0, 1]")
        object.__setattr__(self, "curve_work", _readonly(curve, np.float64))
        object.__setattr__(
            self,
            "signed_distance_field",
            _readonly(distance, np.float64),
        )
        object.__setattr__(
            self,
            "confidence_field",
            _readonly(confidence, np.float64),
        )
        object.__setattr__(
            self,
            "source_labels",
            tuple(str(value) for value in self.source_labels),
        )


@dataclass(frozen=True)
class AlarSurfaceObservationBundle:
    targets: tuple[AlarCurveTarget, ...]
    source: NasalObservationBundle

    def __post_init__(self) -> None:
        if not isinstance(self.source, NasalObservationBundle):
            raise ValueError("source must be a NasalObservationBundle")
        targets = tuple(self.targets)
        if tuple(target.name for target in targets) != ALAR_TARGET_NAMES:
            raise ValueError("alar targets must use canonical ordering")
        object.__setattr__(self, "targets", targets)

    @property
    def by_name(self) -> Mapping[str, AlarCurveTarget]:
        return MappingProxyType({target.name: target for target in self.targets})

    @property
    def by_view(self) -> Mapping[str, tuple[AlarCurveTarget, ...]]:
        return MappingProxyType(
            {
                view: tuple(
                    target for target in self.targets
                    if target.semantic_view == view
                )
                for view in NASAL_VIEWS
            }
        )


def build_alar_surface_observations(
    source: NasalObservationBundle,
    *,
    front_fraction: float = 0.48,
    profile_fraction: float = 0.42,
    target_sample_count: int = 24,
) -> AlarSurfaceObservationBundle:
    """Select outer-wing evidence without introducing landmark targets."""
    if not isinstance(source, NasalObservationBundle):
        raise ValueError("source must be a NasalObservationBundle")
    if not 0.2 <= float(profile_fraction) <= 0.7:
        raise ValueError("profile_fraction must lie in [0.2, 0.7]")
    if not 0.3 <= float(front_fraction) <= 0.7:
        raise ValueError("front_fraction must lie in [0.3, 0.7]")
    if int(target_sample_count) < 12:
        raise ValueError("target_sample_count must be at least 12")

    definitions = (
        (
            "front_subject_left_alar",
            "front",
            "subject-left-alar",
            ("subject_left_nose_wing",),
            1.0,
        ),
        (
            "front_subject_right_alar",
            "front",
            "subject-right-alar",
            ("subject_right_nose_wing",),
            -1.0,
        ),
        (
            "subject_left_alar_profile",
            "subject-left",
            "nasal-profile",
            ("subject_left_nose_wing",),
            -1.0,
        ),
        (
            "subject_right_alar_profile",
            "subject-right",
            "nasal-profile",
            ("subject_right_nose_wing",),
            1.0,
        ),
    )
    targets = []
    for name, view, boundary, labels, fixed_sign in definitions:
        observation = source.by_view[view]
        curve = np.asarray(observation.boundaries_work[boundary], dtype=np.float64)
        if view == "front":
            curve = _lower_alar_segment(curve, float(front_fraction))
        else:
            curve = _profile_alar_segment(
                curve,
                np.asarray(observation.anchors_work["alar_transition"]),
                float(profile_fraction),
            )
        curve = _resample_curve(curve, int(target_sample_count))
        outward_sign = float(fixed_sign)
        if outward_sign == 0.0:
            roi_center = 0.5 * (
                float(observation.roi_work_xyxy[0])
                + float(observation.roi_work_xyxy[2])
            )
            outward_sign = 1.0 if float(np.median(curve[:, 0])) >= roi_center else -1.0
        distance = _rasterized_distance(
            curve,
            observation.distance_field.shape,
            outward_sign,
        )
        targets.append(
            AlarCurveTarget(
                name=name,
                semantic_view=view,
                boundary_name=boundary,
                source_labels=labels,
                curve_work=curve,
                signed_distance_field=distance,
                confidence_field=observation.confidence,
            )
        )
    return AlarSurfaceObservationBundle(tuple(targets), source)


__all__ = [
    "ALAR_TARGET_NAMES",
    "AlarCurveTarget",
    "AlarSurfaceObservationBundle",
    "build_alar_surface_observations",
]

"""Ordered semantic controls for local facial texture registration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import cv2
import numpy as np


@dataclass(frozen=True)
class SemanticFeatureControls:
    model_points: np.ndarray
    observed_points: np.ndarray
    groups: tuple[str, ...]
    confidence: float
    diagnostics: dict[str, Any]

    def group(self, name: str) -> "SemanticFeatureControls":
        keep = np.asarray([group == name for group in self.groups], dtype=bool)
        return SemanticFeatureControls(
            model_points=self.model_points[keep],
            observed_points=self.observed_points[keep],
            groups=tuple(group for group, selected in zip(self.groups, keep) if selected),
            confidence=self.confidence,
            diagnostics=self.diagnostics,
        )


@dataclass(frozen=True)
class NostrilObservations:
    centers: np.ndarray
    confidence: float
    diagnostics: dict[str, Any]


def detect_nostril_observations(
    image: np.ndarray,
    observed_landmarks: np.ndarray,
    nose_mask: np.ndarray,
) -> NostrilObservations:
    """Detect paired dark nostril regions, returning nothing when evidence is weak."""
    rgb = np.asarray(image)
    landmarks = np.asarray(observed_landmarks, dtype=np.float32)
    binary_nose = np.asarray(nose_mask) > 0
    if rgb.ndim != 3 or landmarks.shape[0] < 36 or binary_nose.shape != rgb.shape[:2]:
        return NostrilObservations(
            centers=np.empty((0, 2), dtype=np.float32),
            confidence=0.0,
            diagnostics={"reason": "invalid_inputs"},
        )
    gray = cv2.cvtColor(rgb.astype(np.uint8), cv2.COLOR_RGB2GRAY)
    blackhat = cv2.morphologyEx(
        gray,
        cv2.MORPH_BLACKHAT,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (19, 9)),
    )
    center_x = float(landmarks[33, 0])
    x_min = int(np.floor(min(landmarks[31, 0], center_x)))
    x_max = int(np.ceil(max(landmarks[35, 0], center_x)))
    alar_width = max(float(x_max - x_min), 8.0)
    nose_height = max(float(np.ptp(landmarks[27:36, 1])), 8.0)
    lower_span = max(float(np.max(landmarks[31:36, 1]) - landmarks[30, 1]), 8.0)
    y_min = int(np.floor(landmarks[30, 1] - 0.10 * lower_span))
    y_max = int(np.ceil(np.max(landmarks[31:36, 1]) + 0.12 * lower_span))
    height, width = gray.shape
    roi = np.zeros_like(binary_nose)
    roi[
        max(0, y_min):min(height, y_max + 1),
        max(0, x_min):min(width, x_max + 1),
    ] = True
    roi &= binary_nose
    responses = blackhat[roi]
    positive = responses[responses > 0]
    if len(positive) < 8 or float(positive.max()) < 8.0:
        return NostrilObservations(
            centers=np.empty((0, 2), dtype=np.float32),
            confidence=0.0,
            diagnostics={"reason": "insufficient_dark_response"},
        )
    threshold = max(8.0, float(np.percentile(positive, 72)))
    candidate = roi & (blackhat >= threshold)
    candidate = cv2.morphologyEx(
        candidate.astype(np.uint8),
        cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    ) > 0

    centers: list[np.ndarray] = []
    strengths: list[float] = []
    xx = np.arange(width)[None, :]
    for side_mask in (xx < center_x, xx > center_x):
        side = (candidate & side_mask).astype(np.uint8)
        count, labels, stats, _ = cv2.connectedComponentsWithStats(side, 8)
        best = None
        for label in range(1, count):
            area = int(stats[label, cv2.CC_STAT_AREA])
            if area < 3 or area > max(500, int(roi.sum() * 0.18)):
                continue
            component = labels == label
            values = blackhat[component].astype(np.float64)
            ys, xs = np.where(component)
            weight_sum = float(values.sum())
            if weight_sum <= 1e-6:
                continue
            center = np.array(
                [float(np.sum(xs * values) / weight_sum), float(np.sum(ys * values) / weight_sum)],
                dtype=np.float32,
            )
            minimum_center_offset = 0.12 * alar_width
            if abs(float(center[0]) - center_x) < minimum_center_offset:
                continue
            if float(center[1]) < float(landmarks[30, 1] - 0.06 * lower_span):
                continue
            strength = float(values.mean() * np.sqrt(area))
            if best is None or strength > best[0]:
                best = (strength, center, area)
        if best is None:
            return NostrilObservations(
                centers=np.empty((0, 2), dtype=np.float32),
                confidence=0.0,
                diagnostics={"reason": "paired_components_not_found"},
            )
        strengths.append(best[0])
        centers.append(best[1])

    centers_array = np.vstack(centers).astype(np.float32)
    vertical_delta = abs(float(centers_array[0, 1] - centers_array[1, 1]))
    max_vertical_delta = max(0.55 * lower_span, 4.0)
    if vertical_delta > max_vertical_delta:
        return NostrilObservations(
            centers=np.empty((0, 2), dtype=np.float32),
            confidence=0.0,
            diagnostics={
                "reason": "paired_components_not_level",
                "vertical_delta_px": vertical_delta,
                "max_vertical_delta_px": max_vertical_delta,
            },
        )
    symmetry = float(np.clip(1.0 - vertical_delta / max(max_vertical_delta, 1.0), 0.0, 1.0))
    response_confidence = float(np.clip(min(strengths) / 90.0, 0.0, 1.0))
    confidence = response_confidence * (0.35 + 0.65 * symmetry)
    return NostrilObservations(
        centers=centers_array,
        confidence=float(confidence),
        diagnostics={
            "threshold": threshold,
            "strengths": strengths,
            "vertical_delta_px": vertical_delta,
            "symmetry": symmetry,
        },
    )


def _sample_polyline(points: np.ndarray, count: int) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 2 or len(points) == 0:
        return np.empty((0, 2), dtype=np.float32)
    if len(points) == 1:
        return np.repeat(points, count, axis=0)
    segment_length = np.linalg.norm(np.diff(points, axis=0), axis=1)
    cumulative = np.r_[0.0, np.cumsum(segment_length)]
    if cumulative[-1] <= 1e-6:
        return np.repeat(points[:1], count, axis=0)
    targets = np.linspace(0.0, cumulative[-1], int(count), dtype=np.float32)
    result = np.empty((len(targets), 2), dtype=np.float32)
    for axis in range(2):
        result[:, axis] = np.interp(targets, cumulative, points[:, axis])
    return result


def _lower_mask_curve(
    nose_mask: np.ndarray,
    *,
    x_min: float,
    x_max: float,
    count: int,
) -> np.ndarray:
    binary = np.asarray(nose_mask) > 0
    if binary.ndim != 2 or not np.any(binary) or x_max - x_min < 3.0:
        return np.empty((0, 2), dtype=np.float32)
    height, width = binary.shape
    lo = int(np.clip(np.floor(x_min), 0, width - 1))
    hi = int(np.clip(np.ceil(x_max), 0, width - 1))
    points: list[tuple[float, float]] = []
    for x in range(lo, hi + 1):
        ys = np.flatnonzero(binary[:, x])
        if len(ys):
            points.append((float(x), float(ys.max())))
    if len(points) < max(4, count // 2):
        return np.empty((0, 2), dtype=np.float32)
    curve = np.asarray(points, dtype=np.float32)
    kernel = min(9, len(curve) if len(curve) % 2 else len(curve) - 1)
    if kernel >= 3:
        curve[:, 1] = cv2.GaussianBlur(curve[:, 1, None], (1, kernel), 0).ravel()
    return _sample_polyline(curve, count)


def build_nose_controls(
    projected_landmarks: np.ndarray,
    observed_landmarks: np.ndarray,
    nose_mask: np.ndarray,
    *,
    view: str,
    samples_per_side: int = 6,
    image: np.ndarray | None = None,
) -> SemanticFeatureControls:
    """Build controls with fixed semantics instead of nearest-neighbor matches."""
    projected = np.asarray(projected_landmarks, dtype=np.float32)
    observed = np.asarray(observed_landmarks, dtype=np.float32)
    if projected.shape[0] < 36 or observed.shape[0] < 36:
        raise ValueError("68-point projected and observed landmarks are required")

    if view != "front":
        model = projected[27:36].copy()
        target = observed[27:36].copy()
        return SemanticFeatureControls(
            model_points=model,
            observed_points=target,
            groups=("named_landmarks",) * len(model),
            confidence=0.65,
            diagnostics={
                "matching": "same_landmark_index",
                "boundary_control_count": 0,
            },
        )

    model_parts = [projected[30:31]]
    observed_parts = [observed[30:31]]
    groups: list[str] = ["tip_anchor"]
    center_x = float(observed[33, 0])
    left_outer_x = float(min(observed[31, 0], center_x - 2.0))
    right_outer_x = float(max(observed[35, 0], center_x + 2.0))
    left_observed = _lower_mask_curve(
        nose_mask,
        x_min=left_outer_x,
        x_max=center_x,
        count=samples_per_side,
    )
    right_observed = _lower_mask_curve(
        nose_mask,
        x_min=center_x,
        x_max=right_outer_x,
        count=samples_per_side,
    )
    boundary_count = 0
    if len(left_observed) == samples_per_side and len(right_observed) == samples_per_side:
        left_model = _sample_polyline(projected[[31, 32, 33]], samples_per_side)
        right_model = _sample_polyline(projected[[33, 34, 35]], samples_per_side)
        model_parts.extend((left_model, right_model))
        observed_parts.extend((left_observed, right_observed))
        groups.extend(["lower_left"] * samples_per_side)
        groups.extend(["lower_right"] * samples_per_side)
        boundary_count = samples_per_side * 2

    nostril = NostrilObservations(
        centers=np.empty((0, 2), dtype=np.float32),
        confidence=0.0,
        diagnostics={"reason": "image_not_provided"},
    )
    if image is not None:
        nostril = detect_nostril_observations(image, observed, nose_mask)
    if nostril.confidence >= 0.45 and len(nostril.centers) == 2:
        model_nostrils = np.vstack(
            (
                projected[[30, 31, 32]].mean(axis=0),
                projected[[30, 34, 35]].mean(axis=0),
            )
        ).astype(np.float32)
        model_parts.append(model_nostrils)
        observed_parts.append(nostril.centers)
        groups.extend(("nostril_left", "nostril_right"))

    model = np.vstack(model_parts).astype(np.float32)
    target = np.vstack(observed_parts).astype(np.float32)
    finite = np.isfinite(model).all(axis=1) & np.isfinite(target).all(axis=1)
    model = model[finite]
    target = target[finite]
    groups = [group for group, keep in zip(groups, finite) if keep]
    return SemanticFeatureControls(
        model_points=model,
        observed_points=target,
        groups=tuple(groups),
        confidence=0.9 if boundary_count else 0.3,
        diagnostics={
            "matching": "ordered_lower_boundary",
            "boundary_control_count": int(boundary_count),
            "center_x": center_x,
            "left_outer_x": left_outer_x,
            "right_outer_x": right_outer_x,
            "nostril_confidence": float(nostril.confidence),
            "nostril_diagnostics": nostril.diagnostics,
        },
    )

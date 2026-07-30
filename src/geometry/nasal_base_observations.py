"""View-specific 2-D observations for nasal-base semantic refinement."""

from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import cv2
import numpy as np


NASAL_BASE_ANCHOR_NAMES = (
    "subject_right_outer",
    "subject_right_inner",
    "columella",
    "subject_left_inner",
    "subject_left_outer",
)

_MEDIAPIPE_INDICES = np.asarray((75, 97, 2, 326, 305), dtype=np.int64)
_LANDMARK_68_INDICES = np.asarray((31, 32, 33, 34, 35), dtype=np.int64)
_VIEW_SELECTIONS = {
    "front": np.asarray((0, 1, 2, 3, 4), dtype=np.int64),
    "subject-left": np.asarray((2, 3, 4), dtype=np.int64),
    "subject-right": np.asarray((0, 1, 2), dtype=np.int64),
}

__all__ = [
    "NASAL_BASE_ANCHOR_NAMES",
    "NasalBaseObservationBundle",
    "NasalBaseViewObservation",
    "build_nasal_base_observations",
]


def _readonly_array(value: np.ndarray, dtype=None) -> np.ndarray:
    contiguous = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        contiguous.tobytes(order="C"),
        dtype=contiguous.dtype,
        count=contiguous.size,
    ).reshape(contiguous.shape)


@dataclass(frozen=True)
class NasalBaseViewObservation:
    semantic_view: str
    anchor_names: tuple[str, ...]
    mediapipe_indices: np.ndarray
    landmark_68_indices: np.ndarray
    source_xy: np.ndarray
    target_xy: np.ndarray
    confidence: np.ndarray

    def __post_init__(self) -> None:
        object.__setattr__(self, "semantic_view", str(self.semantic_view))
        object.__setattr__(
            self,
            "anchor_names",
            tuple(str(name) for name in self.anchor_names),
        )
        object.__setattr__(
            self,
            "mediapipe_indices",
            _readonly_array(self.mediapipe_indices, np.int64),
        )
        object.__setattr__(
            self,
            "landmark_68_indices",
            _readonly_array(self.landmark_68_indices, np.int64),
        )
        object.__setattr__(
            self,
            "source_xy",
            _readonly_array(self.source_xy, np.float64),
        )
        object.__setattr__(
            self,
            "target_xy",
            _readonly_array(self.target_xy, np.float64),
        )
        object.__setattr__(
            self,
            "confidence",
            _readonly_array(self.confidence, np.float64),
        )
        count = len(self.anchor_names)
        if self.mediapipe_indices.shape != (count,):
            raise ValueError("mediapipe indices do not match anchor count")
        if self.landmark_68_indices.shape != (count,):
            raise ValueError("68-point indices do not match anchor count")
        if self.source_xy.shape != (count, 2) or self.target_xy.shape != (count, 2):
            raise ValueError("observation coordinates must have shape (N, 2)")
        if self.confidence.shape != (count,):
            raise ValueError("confidence must have shape (N,)")
        if not np.isfinite(self.source_xy).all():
            raise ValueError("source coordinates must be finite")
        if not np.isfinite(self.target_xy).all():
            raise ValueError("target coordinates must be finite")
        if not np.isfinite(self.confidence).all():
            raise ValueError("confidence must be finite")
        if np.any(self.confidence <= 0.0) or np.any(self.confidence > 1.0):
            raise ValueError("confidence must lie in (0, 1]")


@dataclass(frozen=True)
class NasalBaseObservationBundle:
    front: NasalBaseViewObservation
    subject_left: NasalBaseViewObservation
    subject_right: NasalBaseViewObservation

    @property
    def by_view(self) -> Mapping[str, NasalBaseViewObservation]:
        return MappingProxyType(
            {
                "front": self.front,
                "subject-left": self.subject_left,
                "subject-right": self.subject_right,
            }
        )


def _validate_image(image: np.ndarray, semantic_view: str) -> np.ndarray:
    value = np.asarray(image)
    if value.ndim not in (2, 3) or value.shape[0] < 3 or value.shape[1] < 3:
        raise ValueError(f"{semantic_view} image has an invalid shape")
    if value.ndim == 3 and value.shape[2] not in (3, 4):
        raise ValueError(f"{semantic_view} image must have 3 or 4 channels")
    if not np.issubdtype(value.dtype, np.number):
        raise ValueError(f"{semantic_view} image must be numeric")
    if not np.isfinite(value).all():
        raise ValueError(f"{semantic_view} image must be finite")
    return np.array(value, copy=True)


def _validate_landmarks(
    landmarks: np.ndarray,
    semantic_view: str,
    image_shape: tuple[int, ...],
) -> np.ndarray:
    value = np.asarray(landmarks, dtype=np.float64)
    if value.ndim != 2 or value.shape[1] != 2 or len(value) <= 327:
        raise ValueError(f"{semantic_view} landmarks must have shape (N>=328, 2)")
    if not np.isfinite(value).all():
        raise ValueError(f"{semantic_view} landmarks must be finite")
    selected = np.array(value[_MEDIAPIPE_INDICES], copy=True)
    height, width = image_shape[:2]
    if (
        np.any(selected[:, 0] < 0.0)
        or np.any(selected[:, 0] > width - 1)
        or np.any(selected[:, 1] < 0.0)
        or np.any(selected[:, 1] > height - 1)
    ):
        raise ValueError(f"{semantic_view} nasal landmarks lie outside the image")
    return selected


def _grayscale_float(image: np.ndarray) -> np.ndarray:
    if image.ndim == 2:
        gray = image
    elif image.shape[2] == 4:
        gray = cv2.cvtColor(image, cv2.COLOR_BGRA2GRAY)
    else:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray_float = np.asarray(gray, dtype=np.float64)
    minimum = float(np.min(gray_float))
    maximum = float(np.max(gray_float))
    if maximum > minimum:
        gray_float = (gray_float - minimum) / (maximum - minimum)
    else:
        gray_float = np.zeros_like(gray_float)
    return gray_float


def _refine_anchor(
    gray: np.ndarray,
    gradient: np.ndarray,
    source_xy: np.ndarray,
    search_radius_px: int,
) -> tuple[np.ndarray, float]:
    height, width = gray.shape
    x0 = int(round(float(source_xy[0])))
    y0 = int(round(float(source_xy[1])))
    x_min = max(0, x0 - search_radius_px)
    x_max = min(width - 1, x0 + search_radius_px)
    y_min = max(0, y0 - search_radius_px)
    y_max = min(height - 1, y0 + search_radius_px)
    xs = np.arange(x_min, x_max + 1, dtype=np.float64)
    ys = np.arange(y_min, y_max + 1, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys)
    distance = np.sqrt((xx - source_xy[0]) ** 2 + (yy - source_xy[1]) ** 2)
    sigma = max(float(search_radius_px) * 0.55, 1.0)
    prior = np.exp(-(distance**2) / (2.0 * sigma**2))
    local_gradient = gradient[y_min : y_max + 1, x_min : x_max + 1]
    local_darkness = 1.0 - gray[y_min : y_max + 1, x_min : x_max + 1]
    score = prior * (0.55 * local_gradient + 0.45 * local_darkness)
    flat_index = int(np.argmax(score))
    row, column = np.unravel_index(flat_index, score.shape)
    target = np.asarray((xs[column], ys[row]), dtype=np.float64)
    score_peak = float(score[row, column])
    confidence = float(np.clip(0.25 + 0.75 * score_peak, 0.05, 1.0))
    return target, confidence


def build_nasal_base_observations(
    images_by_view: Mapping[str, np.ndarray],
    landmarks_by_view: Mapping[str, np.ndarray],
    *,
    refine_to_image: bool = True,
    search_radius_px: int = 6,
) -> NasalBaseObservationBundle:
    """Build subject-semantic nasal-base anchors in three calibrated views."""
    if isinstance(search_radius_px, (bool, np.bool_)) or not isinstance(
        search_radius_px,
        (int, np.integer),
    ):
        raise ValueError("search_radius_px must be an integer")
    if not 1 <= int(search_radius_px) <= 12:
        raise ValueError("search_radius_px must lie in [1, 12]")
    required_views = tuple(_VIEW_SELECTIONS)
    if any(view not in images_by_view for view in required_views):
        raise ValueError("images must include front, subject-left, and subject-right")
    if any(view not in landmarks_by_view for view in required_views):
        raise ValueError(
            "landmarks must include front, subject-left, and subject-right"
        )

    observations = {}
    for semantic_view in required_views:
        image = _validate_image(images_by_view[semantic_view], semantic_view)
        source_all = _validate_landmarks(
            landmarks_by_view[semantic_view],
            semantic_view,
            image.shape,
        )
        selection = _VIEW_SELECTIONS[semantic_view]
        source = source_all[selection]
        target = source.copy()
        confidence = np.ones(len(selection), dtype=np.float64)
        if refine_to_image:
            gray = _grayscale_float(image)
            grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            gradient = np.hypot(grad_x, grad_y)
            gradient_peak = float(np.max(gradient))
            if gradient_peak > 0.0:
                gradient /= gradient_peak
            for index, point in enumerate(source):
                target[index], confidence[index] = _refine_anchor(
                    gray,
                    gradient,
                    point,
                    int(search_radius_px),
                )
        observations[semantic_view] = NasalBaseViewObservation(
            semantic_view=semantic_view,
            anchor_names=tuple(
                NASAL_BASE_ANCHOR_NAMES[index] for index in selection
            ),
            mediapipe_indices=_MEDIAPIPE_INDICES[selection],
            landmark_68_indices=_LANDMARK_68_INDICES[selection],
            source_xy=source,
            target_xy=target,
            confidence=confidence,
        )

    return NasalBaseObservationBundle(
        front=observations["front"],
        subject_left=observations["subject-left"],
        subject_right=observations["subject-right"],
    )

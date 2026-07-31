"""Coordinate contracts and contour utilities for image observations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import cv2
import numpy as np

from src.cross_view_geometry import (
    Camera,
    relative_camera_transform,
    scale_intrinsics,
)


PixelFrame = Literal["distorted", "undistorted"]


def _readonly_array(
    value: Any,
    *,
    dtype: np.dtype | type = np.float64,
    shape: tuple[int, ...] | None = None,
) -> np.ndarray:
    result = np.array(value, dtype=dtype, copy=True)
    if shape is not None:
        result = result.reshape(shape)
    result.setflags(write=False)
    return result


@dataclass(frozen=True)
class ObservationCoordinates:
    """Describe one view's original pixels and normalized work frame.

    ``pixel_frame`` applies to the image, original-pixel points, and masks
    supplied for this view. Work pixels are always undistorted.
    """

    original_size: tuple[int, int]
    work_size: tuple[int, int]
    K: np.ndarray
    dist: np.ndarray
    pixel_frame: PixelFrame

    def __post_init__(self) -> None:
        original_size = tuple(int(value) for value in self.original_size)
        work_size = tuple(int(value) for value in self.work_size)
        if len(original_size) != 2 or min(original_size) <= 0:
            raise ValueError("original size dimensions must be positive")
        if len(work_size) != 2 or min(work_size) <= 0:
            raise ValueError("work size dimensions must be positive")
        if self.pixel_frame not in {"distorted", "undistorted"}:
            raise ValueError("pixel frame must be distorted or undistorted")
        intrinsics = _readonly_array(self.K, shape=(3, 3))
        distortion = _readonly_array(self.dist).reshape(-1)
        distortion.setflags(write=False)
        if not np.isfinite(intrinsics).all() or not np.isfinite(distortion).all():
            raise ValueError("camera calibration must be finite")
        if (
            self.pixel_frame == "undistorted"
            and np.any(np.abs(distortion) > 1e-12)
        ):
            raise ValueError(
                "undistorted pixel frames must use zero distortion coefficients"
            )
        object.__setattr__(self, "original_size", original_size)
        object.__setattr__(self, "work_size", work_size)
        object.__setattr__(self, "K", intrinsics)
        object.__setattr__(self, "dist", distortion)

    @classmethod
    def from_camera(
        cls,
        camera: Camera,
        *,
        work_size: tuple[int, int],
        pixel_frame: PixelFrame,
    ) -> "ObservationCoordinates":
        return cls(
            original_size=camera.image_size,
            work_size=work_size,
            K=camera.K,
            dist=(
                camera.dist
                if pixel_frame == "distorted"
                else np.zeros_like(np.asarray(camera.dist, dtype=np.float64))
            ),
            pixel_frame=pixel_frame,
        )

    @property
    def work_intrinsics(self) -> np.ndarray:
        result = scale_intrinsics(
            self.K,
            self.original_size,
            self.work_size,
        )
        result.setflags(write=False)
        return result

    def metadata(self) -> dict[str, Any]:
        return {
            "source_pixel_frame": f"{self.pixel_frame}_original_px",
            "mask_layouts": ["original", "square_letterbox"],
            "observation_pixel_frame": "undistorted_work_px",
            "original_size_wh": list(self.original_size),
            "work_size_wh": list(self.work_size),
            "intrinsics": self.K.astype(float).tolist(),
            "distortion_coefficients": self.dist.astype(float).tolist(),
            "undistortion_applied": self.pixel_frame == "distorted",
            "distorted_reverse_available": self.pixel_frame == "distorted",
            "conversion_source": "src.geometry.observation_coordinates",
        }


def letterbox_parameters(
    image_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> tuple[float, int, int, int, int]:
    image_width, image_height = (int(value) for value in image_size)
    canvas_height, canvas_width = (int(value) for value in canvas_shape)
    if min(image_width, image_height, canvas_width, canvas_height) <= 0:
        raise ValueError("image and canvas dimensions must be positive")
    scale = min(
        canvas_width / float(image_width),
        canvas_height / float(image_height),
    )
    resized_width = int(image_width * scale)
    resized_height = int(image_height * scale)
    x_offset = (canvas_width - resized_width) // 2
    y_offset = (canvas_height - resized_height) // 2
    return (
        float(scale),
        int(x_offset),
        int(y_offset),
        int(resized_width),
        int(resized_height),
    )


def intrinsics_to_letterbox_canvas(
    intrinsics: Any,
    source_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    """Map camera intrinsics from source pixels into a letterbox canvas."""
    matrix = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
    if not np.isfinite(matrix).all():
        raise ValueError("camera intrinsics must be finite")
    scale, x_offset, y_offset, _width, _height = letterbox_parameters(
        source_size,
        canvas_shape,
    )
    transform = np.array(
        [
            [scale, 0.0, float(x_offset)],
            [0.0, scale, float(y_offset)],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    return transform @ matrix


def canvas_points_to_original(
    points: Any,
    image_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scale, x_offset, y_offset, _width, _height = letterbox_parameters(
        image_size,
        canvas_shape,
    )
    result = values.copy()
    result[:, 0] = (result[:, 0] - x_offset) / scale
    result[:, 1] = (result[:, 1] - y_offset) / scale
    return result


def original_points_to_canvas(
    points: Any,
    image_size: tuple[int, int],
    canvas_shape: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    scale, x_offset, y_offset, _width, _height = letterbox_parameters(
        image_size,
        canvas_shape,
    )
    result = values * scale
    result[:, 0] += x_offset
    result[:, 1] += y_offset
    return result


def _scale_points(
    points: Any,
    source_size: tuple[int, int],
    target_size: tuple[int, int],
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return values * np.asarray(
        (
            target_size[0] / float(source_size[0]),
            target_size[1] / float(source_size[1]),
        ),
        dtype=np.float64,
    )


def original_points_to_work(
    points: Any,
    coordinates: ObservationCoordinates,
) -> np.ndarray:
    """Map declared original pixels to undistorted work pixels exactly once."""
    scaled = _scale_points(
        points,
        coordinates.original_size,
        coordinates.work_size,
    )
    if coordinates.pixel_frame == "undistorted":
        return scaled
    intrinsics = coordinates.work_intrinsics
    return cv2.undistortPoints(
        scaled.reshape(-1, 1, 2),
        intrinsics,
        coordinates.dist,
        P=intrinsics,
    ).reshape(-1, 2)


def work_points_to_original(
    points: Any,
    coordinates: ObservationCoordinates,
    *,
    target_pixel_frame: PixelFrame | None = None,
) -> np.ndarray:
    """Map work pixels back, rejecting unavailable distorted calibration."""
    target = target_pixel_frame or coordinates.pixel_frame
    if target not in {"distorted", "undistorted"}:
        raise ValueError("target pixel frame must be distorted or undistorted")
    if (
        target == "distorted"
        and coordinates.pixel_frame == "undistorted"
    ):
        raise ValueError(
            "distorted target pixels are unavailable from an undistorted "
            "contract because original distortion was not retained"
        )
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if target == "undistorted":
        return _scale_points(
            values,
            coordinates.work_size,
            coordinates.original_size,
        )
    intrinsics = coordinates.work_intrinsics
    homogeneous = np.column_stack((values, np.ones(len(values))))
    normalized = (np.linalg.inv(intrinsics) @ homogeneous.T).T
    object_points = normalized / normalized[:, 2:3]
    projected, _jacobian = cv2.projectPoints(
        object_points,
        np.zeros(3, dtype=np.float64),
        np.zeros(3, dtype=np.float64),
        intrinsics,
        coordinates.dist,
    )
    return _scale_points(
        projected.reshape(-1, 2),
        coordinates.work_size,
        coordinates.original_size,
    )


def _validate_original_image(
    image: np.ndarray,
    coordinates: ObservationCoordinates,
) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim not in (2, 3):
        raise ValueError("image must be grayscale or have color channels")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError("image must have one, three, or four channels")
    actual_size = (int(frame.shape[1]), int(frame.shape[0]))
    if actual_size != coordinates.original_size:
        raise ValueError(
            f"image size {actual_size} does not match coordinate original "
            f"size {coordinates.original_size}"
        )
    return frame


def normalize_image_to_work(
    image: np.ndarray,
    coordinates: ObservationCoordinates,
) -> np.ndarray:
    frame = _validate_original_image(image, coordinates)
    resized = cv2.resize(
        frame,
        coordinates.work_size,
        interpolation=cv2.INTER_AREA,
    )
    if coordinates.pixel_frame == "undistorted":
        return resized
    intrinsics = coordinates.work_intrinsics
    return cv2.undistort(
        resized,
        intrinsics,
        coordinates.dist,
        None,
        intrinsics,
    )


def normalize_mask_to_work(
    mask: np.ndarray,
    coordinates: ObservationCoordinates,
    *,
    name: str,
) -> np.ndarray:
    """Restore original/letterbox masks and normalize to fixed work pixels."""
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    original_shape = (
        coordinates.original_size[1],
        coordinates.original_size[0],
    )
    if values.shape == original_shape:
        restored = values
    elif values.shape[0] == values.shape[1]:
        (
            _scale,
            x_offset,
            y_offset,
            resized_width,
            resized_height,
        ) = letterbox_parameters(coordinates.original_size, values.shape)
        restored = values[
            y_offset : y_offset + resized_height,
            x_offset : x_offset + resized_width,
        ]
    else:
        raise ValueError(
            f"{name} size {values.shape} must match the original image or "
            "be a square letterbox canvas"
        )
    binary = np.asarray(restored > 0, dtype=np.uint8)
    normalized = cv2.resize(
        binary,
        coordinates.work_size,
        interpolation=cv2.INTER_NEAREST,
    )
    if coordinates.pixel_frame == "distorted":
        intrinsics = coordinates.work_intrinsics
        map_x, map_y = cv2.initUndistortRectifyMap(
            intrinsics,
            coordinates.dist,
            None,
            intrinsics,
            coordinates.work_size,
            cv2.CV_32FC1,
        )
        normalized = cv2.remap(
            normalized,
            map_x,
            map_y,
            interpolation=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    normalized = largest_binary_component(normalized > 0)
    if not np.any(normalized):
        raise ValueError(f"{name} is empty")
    return normalized


def largest_binary_component(mask: np.ndarray) -> np.ndarray:
    binary = np.asarray(mask > 0, dtype=np.uint8)
    count, labels, stats, _centroids = cv2.connectedComponentsWithStats(binary)
    if count <= 1:
        return binary
    largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return np.asarray(labels == largest, dtype=np.uint8)


def mask_variant_work(mask_work: np.ndarray, offset_work_px: int) -> np.ndarray:
    binary = largest_binary_component(mask_work)
    amount = abs(int(offset_work_px))
    if amount == 0:
        return binary.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * amount + 1, 2 * amount + 1),
    )
    operation = cv2.MORPH_DILATE if offset_work_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(binary, operation, kernel)


def resample_polyline_by_arclength(
    points: Any,
    *,
    spacing_px: float = 1.0,
    closed: bool = False,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    if spacing_px <= 0.0:
        raise ValueError("arc-length spacing must be positive")
    if len(values) < 2:
        return values.copy()
    path = np.vstack((values, values[0])) if closed else values
    segment_lengths = np.linalg.norm(np.diff(path, axis=0), axis=1)
    keep = np.concatenate(([True], segment_lengths > 1e-9))
    path = path[keep]
    if len(path) < 2:
        return values[:1].copy()
    cumulative = np.concatenate(
        ([0.0], np.cumsum(np.linalg.norm(np.diff(path, axis=0), axis=1)))
    )
    total = float(cumulative[-1])
    if total <= 1e-9:
        return values[:1].copy()
    sample_count = max(2, int(np.ceil(total / spacing_px)))
    samples = np.linspace(
        0.0,
        total,
        sample_count,
        endpoint=not closed,
    )
    result = np.column_stack(
        (
            np.interp(samples, cumulative, path[:, 0]),
            np.interp(samples, cumulative, path[:, 1]),
        )
    )
    return result.astype(np.float64)


def external_contour_work(
    mask_work: np.ndarray,
    *,
    name: str,
    spacing_work_px: float = 1.0,
) -> np.ndarray:
    binary = largest_binary_component(mask_work)
    contours, _hierarchy = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError(f"{name} has no external contour")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 4:
        raise ValueError(f"{name} contour has too little support")
    return resample_polyline_by_arclength(
        contour,
        spacing_px=spacing_work_px,
        closed=True,
    )


def contour_curvature(points: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    count = len(values)
    if count < 7:
        return np.zeros(count, dtype=np.float64)
    stride = max(2, min(12, count // 150))
    previous = values[np.arange(count) - stride]
    following = values[(np.arange(count) + stride) % count]
    first = values - previous
    second = following - values
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1e-9)
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1e-9)
    return np.clip(1.0 - np.sum(first * second, axis=1), 0.0, 2.0)


def fundamental_matrix_work(
    front_camera: Camera,
    side_camera: Camera,
    front_coordinates: ObservationCoordinates,
    side_coordinates: ObservationCoordinates,
) -> np.ndarray:
    if front_coordinates.work_size != side_coordinates.work_size:
        raise ValueError("epipolar views must share one work size")
    rotation, translation = relative_camera_transform(front_camera, side_camera)
    tx, ty, tz = np.asarray(translation, dtype=np.float64).reshape(3)
    skew = np.array(
        ((0.0, -tz, ty), (tz, 0.0, -tx), (-ty, tx, 0.0)),
        dtype=np.float64,
    )
    return (
        np.linalg.inv(side_coordinates.work_intrinsics).T
        @ skew
        @ rotation
        @ np.linalg.inv(front_coordinates.work_intrinsics)
    )


def point_line_distances(points: np.ndarray, line: np.ndarray) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    a, b, c = np.asarray(line, dtype=np.float64).reshape(3)
    denominator = float(np.hypot(a, b))
    if denominator <= 1e-12:
        return np.full(len(values), np.inf, dtype=np.float64)
    return np.abs(values[:, 0] * a + values[:, 1] * b + c) / denominator

"""Confidence-weighted nasal boundaries for the fixed three-camera rig."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

import cv2
import numpy as np

from src.cross_view_geometry import Camera, scale_intrinsics
from src.geometry.profile_silhouette_extrema import (
    _contour_curvature,
    _fundamental_matrix_work,
    _largest_binary_component,
    _original_to_work,
    _point_line_distances,
    _work_to_original,
    _undistort_work_points,
    canvas_points_to_original,
    original_points_to_canvas,
)
from src.geometry.profile_triangulation import ProfileRig


CAMERA_SEMANTICS = {
    "camera1": ("left", "subject-left"),
    "camera2": ("front", "front"),
    "camera3": ("right", "subject-right"),
}
NASAL_VIEWS = ("front", "subject-left", "subject-right")
PROFILE_PRIOR_NAMES = (
    "upper_tip",
    "tip_apex",
    "lower_tip",
    "alar_transition",
)


@dataclass(frozen=True)
class NasalObservationConfig:
    work_size: tuple[int, int] = (640, 480)
    mask_perturbation_px: int = 2
    gradient_window_px: int = 5
    front_alar_vertical_fraction: tuple[float, float] = (0.45, 0.94)
    profile_prior_padding_px: float = 14.0
    max_epipolar_distance_px: float = 5.0
    epipolar_samples_per_interval: int = 8
    distance_clip_px: float = 48.0
    confidence_spread_px: float = 3.0
    min_boundary_points: int = 12

    def __post_init__(self) -> None:
        width, height = self.work_size
        if width <= 0 or height <= 0:
            raise ValueError("work size dimensions must be positive")
        if self.mask_perturbation_px <= 0:
            raise ValueError("mask perturbation must be positive")
        if self.gradient_window_px <= 0:
            raise ValueError("gradient window must be positive")
        lower, upper = self.front_alar_vertical_fraction
        if not 0.0 <= lower < upper <= 1.0:
            raise ValueError("front alar vertical fractions must satisfy 0 <= low < high <= 1")
        if self.profile_prior_padding_px <= 0.0:
            raise ValueError("profile prior padding must be positive")
        if self.max_epipolar_distance_px <= 0.0:
            raise ValueError("maximum epipolar distance must be positive")
        if self.epipolar_samples_per_interval < 2:
            raise ValueError("epipolar samples per interval must be at least two")
        if self.distance_clip_px <= 0.0:
            raise ValueError("distance clip must be positive")
        if self.confidence_spread_px <= 0.0:
            raise ValueError("confidence spread must be positive")
        if self.min_boundary_points < 2:
            raise ValueError("minimum boundary point count must be at least two")


@dataclass(frozen=True)
class NasalViewObservation:
    """One view; named distance fields are authoritative for optimization."""

    semantic_view: str
    camera: Camera
    original_size: tuple[int, int]
    mask_canvas_shape: tuple[int, int]
    work_size: tuple[int, int]
    roi_work_xyxy: tuple[float, float, float, float]
    boundaries_work: Mapping[str, np.ndarray]
    boundary: np.ndarray
    distance_fields: Mapping[str, np.ndarray]
    distance_field: np.ndarray
    confidence: np.ndarray
    variant_boundaries_work: Mapping[str, Mapping[str, np.ndarray]]
    variant_boundaries: Mapping[str, np.ndarray]
    anchors_work: Mapping[str, np.ndarray] = field(default_factory=dict)
    camera_metadata: Mapping[str, Any] = field(default_factory=dict)
    coordinate_metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.semantic_view not in NASAL_VIEWS:
            raise ValueError(f"unsupported nasal view: {self.semantic_view}")
        height, width = self.boundary.shape
        if (width, height) != self.work_size:
            raise ValueError("boundary size does not match observation work size")
        if self.distance_field.shape != self.boundary.shape:
            raise ValueError("distance field size does not match boundary")
        if set(self.distance_fields) != set(self.boundaries_work):
            raise ValueError("named distance fields must match named boundaries")
        for name, distance_field in self.distance_fields.items():
            if distance_field.shape != self.boundary.shape:
                raise ValueError(
                    f"distance field size does not match boundary for {name}"
                )
            if not np.isfinite(distance_field).all():
                raise ValueError(
                    f"distance field contains non-finite values for {name}"
                )
        if self.confidence.shape != self.boundary.shape:
            raise ValueError("confidence size does not match boundary")
        if "base" not in self.variant_boundaries:
            raise ValueError("variant boundaries must include base")
        if set(self.variant_boundaries) != set(self.variant_boundaries_work):
            raise ValueError("variant curve and raster names must match")
        for name, variant in self.variant_boundaries.items():
            if variant.shape != self.boundary.shape:
                raise ValueError(
                    f"variant boundary size does not match boundary for {name}"
                )
        if not np.isfinite(self.distance_field).all():
            raise ValueError("distance field contains non-finite values")
        if not np.isfinite(self.confidence).all():
            raise ValueError("confidence contains non-finite values")
        if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
            raise ValueError("confidence must stay within [0, 1]")


@dataclass(frozen=True)
class NasalObservationBundle:
    front: NasalViewObservation
    subject_left: NasalViewObservation
    subject_right: NasalViewObservation

    def __post_init__(self) -> None:
        observations = self.by_view
        for view in NASAL_VIEWS:
            observation = observations[view]
            if observation.semantic_view != view:
                raise ValueError(
                    f"bundle field {view} contains {observation.semantic_view}"
                )
            mapped = nasal_view_for_camera(observation.camera)
            if mapped != view:
                raise ValueError(
                    f"bundle camera {observation.camera.name} maps to {mapped}, not {view}"
                )
        sizes = {observation.work_size for observation in observations.values()}
        if len(sizes) != 1:
            raise ValueError("bundle observations must share one work size")

    @property
    def by_view(self) -> dict[str, NasalViewObservation]:
        return {
            "front": self.front,
            "subject-left": self.subject_left,
            "subject-right": self.subject_right,
        }

    @property
    def camera_name_by_view(self) -> dict[str, str]:
        return {
            view: observation.camera.name
            for view, observation in self.by_view.items()
        }


def nasal_view_for_camera(camera: Camera) -> str:
    """Return subject-relative semantics for the project's fixed camera names."""
    definition = CAMERA_SEMANTICS.get(camera.name)
    if definition is None:
        raise ValueError(f"unknown fixed-rig camera: {camera.name}")
    expected_camera_view, nasal_view = definition
    if camera.view != expected_camera_view:
        raise ValueError(
            f"{camera.name} must have camera view {expected_camera_view}, "
            f"found {camera.view}"
        )
    return nasal_view


def original_points_to_work(
    points: Any,
    image_size: tuple[int, int],
    work_size: tuple[int, int],
) -> np.ndarray:
    """Delegate original-to-work scaling to the profile observation geometry."""
    return _original_to_work(points, image_size, work_size)


def work_points_to_original(
    points: Any,
    image_size: tuple[int, int],
    work_size: tuple[int, int],
) -> np.ndarray:
    """Delegate work-to-original scaling to the profile observation geometry."""
    return _work_to_original(points, image_size, work_size)


def _validate_image(image: np.ndarray, camera: Camera) -> np.ndarray:
    frame = np.asarray(image)
    if frame.ndim not in (2, 3):
        raise ValueError("image must be grayscale or have color channels")
    if frame.ndim == 3 and frame.shape[2] not in (1, 3, 4):
        raise ValueError("image must have one, three, or four channels")
    actual_size = (int(frame.shape[1]), int(frame.shape[0]))
    if actual_size != tuple(camera.image_size):
        raise ValueError(
            f"image size {actual_size} does not match {camera.name} "
            f"size {camera.image_size}"
        )
    return frame


def _validate_mask(
    mask: np.ndarray,
    camera: Camera,
    *,
    name: str,
) -> np.ndarray:
    values = np.asarray(mask)
    if values.ndim == 3:
        values = values[..., 0]
    if values.ndim != 2:
        raise ValueError(f"{name} must be a two-dimensional mask")
    original_shape = (camera.image_size[1], camera.image_size[0])
    if values.shape != original_shape and values.shape[0] != values.shape[1]:
        raise ValueError(
            f"{name} size {values.shape} must match the original image or "
            "be a square letterbox canvas"
        )
    binary = _largest_binary_component(values)
    if not np.any(binary):
        raise ValueError(f"{name} is empty")
    return binary


def _mask_variant(binary: np.ndarray, offset_px: int) -> np.ndarray:
    amount = abs(int(offset_px))
    if amount == 0:
        return binary.copy()
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE,
        (2 * amount + 1, 2 * amount + 1),
    )
    operation = cv2.MORPH_DILATE if offset_px > 0 else cv2.MORPH_ERODE
    return cv2.morphologyEx(binary, operation, kernel)


def _external_contour_canvas(binary: np.ndarray, *, name: str) -> np.ndarray:
    contours, _hierarchy = cv2.findContours(
        np.asarray(binary > 0, dtype=np.uint8),
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_NONE,
    )
    if not contours:
        raise ValueError(f"{name} has no external contour")
    contour = max(contours, key=cv2.contourArea).reshape(-1, 2)
    if len(contour) < 4:
        raise ValueError(f"{name} contour has too little support")
    return contour.astype(np.float64)


def _canvas_contour_to_work(
    contour_canvas: np.ndarray,
    camera: Camera,
    canvas_shape: tuple[int, int],
    work_size: tuple[int, int],
) -> np.ndarray:
    original = canvas_points_to_original(
        contour_canvas,
        camera.image_size,
        canvas_shape,
    )
    width, height = camera.image_size
    valid = (
        (original[:, 0] >= 0.0)
        & (original[:, 0] < width)
        & (original[:, 1] >= 0.0)
        & (original[:, 1] < height)
    )
    original = original[valid]
    if len(original) < 4:
        raise ValueError("mask contour has too little support inside image bounds")
    distorted_work = _original_to_work(
        original,
        camera.image_size,
        work_size,
    )
    return _undistort_work_points(distorted_work, camera, work_size)


def _rasterize_curves(
    curves: Mapping[str, np.ndarray],
    work_size: tuple[int, int],
) -> np.ndarray:
    width, height = work_size
    raster = np.zeros((height, width), dtype=np.uint8)
    for points in curves.values():
        values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
        if not len(values):
            continue
        pixels = np.rint(values).astype(np.int32)
        pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
        pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
        if len(pixels) == 1:
            raster[pixels[0, 1], pixels[0, 0]] = 1
        else:
            cv2.polylines(
                raster,
                [pixels.reshape(-1, 1, 2)],
                False,
                1,
                1,
                cv2.LINE_8,
            )
    return raster > 0


def _unsigned_distance_field(
    boundary: np.ndarray,
    distance_clip_px: float,
) -> np.ndarray:
    if not np.any(boundary):
        raise ValueError("cannot build a distance field from an empty boundary")
    background = np.asarray(~boundary, dtype=np.uint8)
    distance = cv2.distanceTransform(background, cv2.DIST_L2, 5)
    return np.minimum(distance, float(distance_clip_px)).astype(np.float32)


def _work_gradients(
    image: np.ndarray,
    camera: Camera,
    config: NasalObservationConfig,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = image
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[..., 0]
    if frame.ndim == 3 and frame.shape[2] == 4:
        frame = cv2.cvtColor(frame, cv2.COLOR_BGRA2BGR)
    if frame.ndim == 3:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    else:
        gray = frame
    gray_work = cv2.resize(
        gray,
        config.work_size,
        interpolation=cv2.INTER_AREA,
    )
    intrinsics = scale_intrinsics(
        camera.K,
        camera.image_size,
        config.work_size,
    )
    gray_work = cv2.undistort(
        gray_work,
        intrinsics,
        camera.dist,
        None,
        intrinsics,
    )
    float_gray = gray_work.astype(np.float32)
    gx = cv2.Sobel(float_gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(float_gray, cv2.CV_32F, 0, 1, ksize=3)
    window = max(1, int(config.gradient_window_px))
    if window % 2 == 0:
        window += 1
    gx = cv2.GaussianBlur(gx, (window, window), 0)
    gy = cv2.GaussianBlur(gy, (window, window), 0)
    magnitude = cv2.magnitude(gx, gy)
    positive = magnitude[magnitude > 1e-6]
    if not len(positive):
        zeros = np.zeros_like(magnitude, dtype=np.float32)
        return zeros, zeros.copy(), zeros.copy()
    normalizer = max(float(np.percentile(positive, 90.0)), 1e-6)
    unit_x = np.divide(
        gx,
        magnitude,
        out=np.zeros_like(gx),
        where=magnitude > 1e-6,
    )
    unit_y = np.divide(
        gy,
        magnitude,
        out=np.zeros_like(gy),
        where=magnitude > 1e-6,
    )
    strength = np.clip(magnitude / normalizer, 0.0, 1.0)
    return (
        unit_x.astype(np.float32),
        unit_y.astype(np.float32),
        strength.astype(np.float32),
    )


def _curve_normal_field(
    curves: Mapping[str, np.ndarray],
    work_size: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray]:
    width, height = work_size
    normal_x = np.zeros((height, width), dtype=np.float32)
    normal_y = np.zeros((height, width), dtype=np.float32)
    counts = np.zeros((height, width), dtype=np.float32)
    for raw_points in curves.values():
        points = np.asarray(raw_points, dtype=np.float64).reshape(-1, 2)
        if len(points) < 2:
            continue
        tangents = np.empty_like(points)
        tangents[0] = points[1] - points[0]
        tangents[-1] = points[-1] - points[-2]
        if len(points) > 2:
            tangents[1:-1] = points[2:] - points[:-2]
        normals = np.column_stack((-tangents[:, 1], tangents[:, 0]))
        norms = np.linalg.norm(normals, axis=1, keepdims=True)
        normals = np.divide(
            normals,
            norms,
            out=np.zeros_like(normals),
            where=norms > 1e-9,
        )
        for index in range(len(points) - 1):
            segment = points[index + 1] - points[index]
            sample_count = max(2, int(np.ceil(np.linalg.norm(segment))) + 1)
            for alpha in np.linspace(0.0, 1.0, sample_count):
                point = points[index] + alpha * segment
                normal = (
                    (1.0 - alpha) * normals[index]
                    + alpha * normals[index + 1]
                )
                normal_norm = float(np.linalg.norm(normal))
                if normal_norm <= 1e-9:
                    continue
                x, y = np.rint(point).astype(np.int32)
                if 0 <= x < width and 0 <= y < height:
                    normal_x[y, x] += float(normal[0] / normal_norm)
                    normal_y[y, x] += float(normal[1] / normal_norm)
                    counts[y, x] += 1.0
    normal_x = np.divide(
        normal_x,
        counts,
        out=np.zeros_like(normal_x),
        where=counts > 0.0,
    )
    normal_y = np.divide(
        normal_y,
        counts,
        out=np.zeros_like(normal_y),
        where=counts > 0.0,
    )
    norm = cv2.magnitude(normal_x, normal_y)
    normal_x = np.divide(
        normal_x,
        norm,
        out=np.zeros_like(normal_x),
        where=norm > 1e-6,
    )
    normal_y = np.divide(
        normal_y,
        norm,
        out=np.zeros_like(normal_y),
        where=norm > 1e-6,
    )
    return normal_x, normal_y


def _gradient_normal_agreement(
    image: np.ndarray,
    camera: Camera,
    curves: Mapping[str, np.ndarray],
    config: NasalObservationConfig,
) -> np.ndarray:
    gradient_x, gradient_y, strength = _work_gradients(
        image,
        camera,
        config,
    )
    normal_x, normal_y = _curve_normal_field(curves, config.work_size)
    alignment = np.abs(
        gradient_x * normal_x + gradient_y * normal_y
    )
    return np.clip(strength * alignment, 0.0, 1.0).astype(np.float32)


def _confidence_field(
    boundary: np.ndarray,
    variant_boundaries: list[np.ndarray],
    distance_field: np.ndarray,
    gradient_agreement: np.ndarray,
    config: NasalObservationConfig,
) -> np.ndarray:
    distances = []
    for variant in variant_boundaries:
        if np.any(variant):
            distances.append(
                cv2.distanceTransform(
                    np.asarray(~variant, dtype=np.uint8),
                    cv2.DIST_L2,
                    5,
                )
            )
        else:
            distances.append(
                np.full(boundary.shape, config.distance_clip_px, dtype=np.float32)
            )
    mean_variant_distance = np.mean(np.asarray(distances), axis=0)
    stability_scale = float(config.mask_perturbation_px) + 0.75
    stability = np.exp(
        -0.5 * np.square(mean_variant_distance / stability_scale)
    ).astype(np.float32)
    boundary_values = np.clip(
        0.20 + 0.60 * stability + 0.20 * gradient_agreement,
        0.0,
        1.0,
    )
    seeds = np.where(boundary, boundary_values, 0.0).astype(np.float32)
    support = boundary.astype(np.float32)
    sigma = float(config.confidence_spread_px)
    numerator = cv2.GaussianBlur(seeds, (0, 0), sigmaX=sigma, sigmaY=sigma)
    denominator = cv2.GaussianBlur(
        support,
        (0, 0),
        sigmaX=sigma,
        sigmaY=sigma,
    )
    propagated = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 1e-8,
    )
    falloff = np.exp(
        -distance_field / max(0.5 * float(config.distance_clip_px), 1e-6)
    )
    confidence = propagated * falloff
    confidence[distance_field >= float(config.distance_clip_px)] = 0.0
    return np.clip(confidence, 0.0, 1.0).astype(np.float32)


def _coordinate_metadata(
    camera: Camera,
    canvas_shape: tuple[int, int],
    work_size: tuple[int, int],
) -> dict[str, Any]:
    return {
        "source_pixel_frame": "distorted_original_px",
        "mask_pixel_frame": "letterbox_canvas_px",
        "observation_pixel_frame": "undistorted_work_px",
        "distance_field": "unsigned_truncated_euclidean_px",
        "aggregate_distance_field_usage": "display_only",
        "original_size_wh": list(camera.image_size),
        "mask_canvas_shape_hw": list(canvas_shape),
        "work_size_wh": list(work_size),
        "conversion_source": (
            "src.geometry.profile_silhouette_extrema"
        ),
    }


def _camera_metadata(camera: Camera, semantic_view: str) -> dict[str, Any]:
    return {
        "camera_name": camera.name,
        "camera_view": camera.view,
        "subject_relative_view": semantic_view,
        "image_size_wh": list(camera.image_size),
        "intrinsics": np.asarray(camera.K, dtype=float).tolist(),
        "distortion_coefficients": np.asarray(
            camera.dist,
            dtype=float,
        ).reshape(-1).tolist(),
        "rig_to_camera_rotation": np.asarray(
            camera.R_rig_to_camera,
            dtype=float,
        ).tolist(),
        "rig_to_camera_translation": np.asarray(
            camera.t_rig_to_camera,
            dtype=float,
        ).reshape(3).tolist(),
    }


def _make_observation(
    *,
    semantic_view: str,
    camera: Camera,
    canvas_shape: tuple[int, int],
    roi_work_xyxy: tuple[float, float, float, float],
    curves: Mapping[str, np.ndarray],
    variant_curves: Mapping[str, Mapping[str, np.ndarray]],
    anchors: Mapping[str, np.ndarray],
    image: np.ndarray,
    config: NasalObservationConfig,
    coordinate_extras: Mapping[str, Any] | None = None,
) -> NasalViewObservation:
    normalized_variants = {
        name: {
            boundary_name: np.asarray(points, dtype=np.float64)
            for boundary_name, points in candidate.items()
        }
        for name, candidate in variant_curves.items()
    }
    if "base" not in normalized_variants:
        normalized_variants["base"] = {
            name: np.asarray(points, dtype=np.float64)
            for name, points in curves.items()
        }
    variant_boundaries = {
        name: _rasterize_curves(candidate, config.work_size)
        for name, candidate in normalized_variants.items()
    }
    boundary = variant_boundaries["base"]
    distance_fields = {
        name: _unsigned_distance_field(
            _rasterize_curves({name: points}, config.work_size),
            config.distance_clip_px,
        )
        for name, points in curves.items()
    }
    distance_field = np.minimum.reduce(
        list(distance_fields.values())
    ).astype(np.float32)
    confidence = _confidence_field(
        boundary,
        [
            candidate
            for name, candidate in variant_boundaries.items()
            if name != "base"
        ],
        distance_field,
        _gradient_normal_agreement(image, camera, curves, config),
        config,
    )
    coordinate_metadata = _coordinate_metadata(
        camera,
        canvas_shape,
        config.work_size,
    )
    if coordinate_extras:
        coordinate_metadata.update(dict(coordinate_extras))
    return NasalViewObservation(
        semantic_view=semantic_view,
        camera=camera,
        original_size=tuple(camera.image_size),
        mask_canvas_shape=tuple(canvas_shape),
        work_size=tuple(config.work_size),
        roi_work_xyxy=tuple(float(value) for value in roi_work_xyxy),
        boundaries_work={
            name: np.asarray(points, dtype=np.float64)
            for name, points in curves.items()
        },
        boundary=boundary,
        distance_fields=distance_fields,
        distance_field=distance_field,
        confidence=confidence,
        variant_boundaries_work=normalized_variants,
        variant_boundaries=variant_boundaries,
        anchors_work={
            name: np.asarray(point, dtype=np.float64).reshape(2)
            for name, point in anchors.items()
        },
        camera_metadata=_camera_metadata(camera, semantic_view),
        coordinate_metadata=coordinate_metadata,
    )


def _front_alar_curves(
    binary: np.ndarray,
    camera: Camera,
    centerline_x_original: float,
    config: NasalObservationConfig,
) -> dict[str, np.ndarray]:
    contour_canvas = _external_contour_canvas(binary, name="nose mask")
    contour_work = _canvas_contour_to_work(
        contour_canvas,
        camera,
        binary.shape,
        config.work_size,
    )
    center_y_original = 0.5 * camera.image_size[1]
    center_work = _undistort_work_points(
        _original_to_work(
            ((centerline_x_original, center_y_original),),
            camera.image_size,
            config.work_size,
        ),
        camera,
        config.work_size,
    )[0, 0]
    y_min = float(np.min(contour_work[:, 1]))
    y_max = float(np.max(contour_work[:, 1]))
    lower_fraction, upper_fraction = config.front_alar_vertical_fraction
    lower_y = y_min + lower_fraction * (y_max - y_min)
    upper_y = y_min + upper_fraction * (y_max - y_min)
    local = contour_work[
        (contour_work[:, 1] >= lower_y)
        & (contour_work[:, 1] <= upper_y)
    ]
    rows = np.rint(local[:, 1]).astype(np.int32)
    subject_left = []
    subject_right = []
    for row in np.unique(rows):
        row_points = local[rows == row]
        left_candidates = row_points[row_points[:, 0] > center_work]
        right_candidates = row_points[row_points[:, 0] < center_work]
        if len(left_candidates):
            subject_left.append(
                left_candidates[int(np.argmax(left_candidates[:, 0]))]
            )
        if len(right_candidates):
            subject_right.append(
                right_candidates[int(np.argmin(right_candidates[:, 0]))]
            )
    curves = {
        "subject-left-alar": np.asarray(subject_left, dtype=np.float64).reshape(-1, 2),
        "subject-right-alar": np.asarray(subject_right, dtype=np.float64).reshape(-1, 2),
    }
    for name, points in curves.items():
        if len(points) < config.min_boundary_points:
            raise ValueError(f"nose mask has too little support for {name}")
    return curves


def build_front_nasal_observation(
    image: np.ndarray,
    semantic_nose_mask: np.ndarray,
    camera: Camera,
    *,
    centerline_x_original: float,
    config: NasalObservationConfig | None = None,
) -> NasalViewObservation:
    """Extract subject-left/right alar evidence from a semantic nose mask."""
    limits = config or NasalObservationConfig()
    frame = _validate_image(image, camera)
    if nasal_view_for_camera(camera) != "front":
        raise ValueError("front nasal observation requires camera2/front")
    centerline = float(centerline_x_original)
    if not np.isfinite(centerline) or not 0.0 <= centerline < camera.image_size[0]:
        raise ValueError("front nasal centerline must lie inside image bounds")
    binary = _validate_mask(
        semantic_nose_mask,
        camera,
        name="nose mask",
    )
    curves = _front_alar_curves(binary, camera, centerline, limits)
    variant_curves: dict[str, Mapping[str, np.ndarray]] = {
        "base": curves,
    }
    for name, offset in (
        ("eroded", -limits.mask_perturbation_px),
        ("dilated", limits.mask_perturbation_px),
    ):
        variant = _mask_variant(binary, offset)
        try:
            variant_curves[name] = _front_alar_curves(
                variant,
                camera,
                centerline,
                limits,
            )
        except ValueError:
            variant_curves[name] = {}
    all_points = np.vstack(list(curves.values()))
    roi = (
        float(np.min(all_points[:, 0])),
        float(np.min(all_points[:, 1])),
        float(np.max(all_points[:, 0]) + 1.0),
        float(np.max(all_points[:, 1]) + 1.0),
    )
    return _make_observation(
        semantic_view="front",
        camera=camera,
        canvas_shape=binary.shape,
        roi_work_xyxy=roi,
        curves=curves,
        variant_curves=variant_curves,
        anchors={},
        image=frame,
        config=limits,
    )


def _validate_profile_roi(
    roi_original_xyxy: tuple[float, float, float, float],
    camera: Camera,
) -> tuple[float, float, float, float]:
    roi = np.asarray(roi_original_xyxy, dtype=np.float64).reshape(4)
    if not np.isfinite(roi).all():
        raise ValueError("profile ROI must contain finite coordinates")
    x0, y0, x1, y1 = (float(value) for value in roi)
    width, height = camera.image_size
    if x0 < 0.0 or y0 < 0.0 or x1 > width or y1 > height:
        raise ValueError("profile ROI must lie inside image bounds")
    if x1 <= x0 or y1 <= y0:
        raise ValueError("profile ROI must have positive width and height")
    return x0, y0, x1, y1


def _named_points_work(
    points_original: Mapping[str, Any],
    camera: Camera,
    config: NasalObservationConfig,
    *,
    label: str,
) -> dict[str, np.ndarray]:
    missing = [
        name for name in PROFILE_PRIOR_NAMES
        if name not in points_original
    ]
    if missing:
        raise ValueError(
            f"{label} is missing: " + ", ".join(missing)
        )
    original = np.asarray(
        [points_original[name] for name in PROFILE_PRIOR_NAMES],
        dtype=np.float64,
    ).reshape(-1, 2)
    if not np.isfinite(original).all():
        raise ValueError(f"{label} points must be finite")
    width, height = camera.image_size
    if np.any(
        (original[:, 0] < 0.0)
        | (original[:, 0] >= width)
        | (original[:, 1] < 0.0)
        | (original[:, 1] >= height)
    ):
        raise ValueError(f"{label} points must lie inside image bounds")
    work = _undistort_work_points(
        _original_to_work(original, camera.image_size, config.work_size),
        camera,
        config.work_size,
    )
    return {
        name: work[index]
        for index, name in enumerate(PROFILE_PRIOR_NAMES)
    }


def _roi_original_to_work(
    roi_original_xyxy: tuple[float, float, float, float],
    camera: Camera,
    work_size: tuple[int, int],
) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = roi_original_xyxy
    corners = np.asarray(
        ((x0, y0), (x1, y0), (x1, y1), (x0, y1)),
        dtype=np.float64,
    )
    work = _undistort_work_points(
        _original_to_work(corners, camera.image_size, work_size),
        camera,
        work_size,
    )
    return (
        float(np.min(work[:, 0])),
        float(np.min(work[:, 1])),
        float(np.max(work[:, 0])),
        float(np.max(work[:, 1])),
    )


def _circular_true_runs(mask: np.ndarray) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if not len(indices):
        return []
    runs = [
        run
        for run in np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        if len(run)
    ]
    if (
        len(runs) > 1
        and runs[0][0] == 0
        and runs[-1][-1] == len(mask) - 1
    ):
        merged = np.concatenate((runs[-1], runs[0]))
        runs = [merged, *runs[1:-1]]
    return runs


def _epipolar_lines_work(
    front_anchors_work: Mapping[str, np.ndarray],
    front_camera: Camera,
    side_camera: Camera,
    config: NasalObservationConfig,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    fundamental = _fundamental_matrix_work(
        front_camera,
        side_camera,
        config.work_size,
    )
    anchor_values = np.asarray(
        [front_anchors_work[name] for name in PROFILE_PRIOR_NAMES],
        dtype=np.float64,
    )
    named_lines = {
        name: fundamental @ np.append(anchor_values[index], 1.0)
        for index, name in enumerate(PROFILE_PRIOR_NAMES)
    }
    samples = []
    count = int(config.epipolar_samples_per_interval)
    for index in range(len(anchor_values) - 1):
        for alpha in np.linspace(0.0, 1.0, count, endpoint=False):
            samples.append(
                (1.0 - alpha) * anchor_values[index]
                + alpha * anchor_values[index + 1]
            )
    samples.append(anchor_values[-1])
    sampled_lines = np.asarray(
        [
            fundamental @ np.append(point, 1.0)
            for point in samples
        ],
        dtype=np.float64,
    )
    norms = np.linalg.norm(sampled_lines[:, :2], axis=1)
    if np.any(~np.isfinite(sampled_lines)) or np.any(norms <= 1e-12):
        raise ValueError("fixed rig produced invalid epipolar lines")
    return named_lines, sampled_lines


def _line_distance_matrix(
    points: np.ndarray,
    lines: np.ndarray,
) -> np.ndarray:
    values = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    line_values = np.asarray(lines, dtype=np.float64).reshape(-1, 3)
    numerators = np.abs(
        values @ line_values[:, :2].T + line_values[None, :, 2]
    )
    denominators = np.linalg.norm(line_values[:, :2], axis=1)
    return numerators / np.maximum(denominators[None, :], 1e-12)


def _ordered_epipolar_candidate(
    points: np.ndarray,
    named_lines: Mapping[str, np.ndarray],
    config: NasalObservationConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray] | None:
    curve = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    anchor_indices = np.asarray(
        [
            int(np.argmin(_point_line_distances(curve, named_lines[name])))
            for name in PROFILE_PRIOR_NAMES
        ],
        dtype=np.int32,
    )
    anchor_errors = np.asarray(
        [
            float(
                _point_line_distances(
                    curve[anchor_indices[index]].reshape(1, 2),
                    named_lines[name],
                )[0]
            )
            for index, name in enumerate(PROFILE_PRIOR_NAMES)
        ],
        dtype=np.float64,
    )
    if np.any(anchor_errors > float(config.max_epipolar_distance_px)):
        return None
    if np.all(np.diff(anchor_indices) <= 0):
        curve = curve[::-1].copy()
        anchor_indices = len(curve) - 1 - anchor_indices
    if np.any(np.diff(anchor_indices) < 0):
        return None
    start = int(anchor_indices[0])
    stop = int(anchor_indices[-1]) + 1
    curve = curve[start:stop]
    anchor_indices = anchor_indices - start
    if len(curve) < config.min_boundary_points:
        return None
    anchors = {
        name: curve[int(anchor_indices[index])]
        for index, name in enumerate(PROFILE_PRIOR_NAMES)
    }
    return curve, anchors, anchor_errors


def _profile_curve_from_epipolar(
    binary: np.ndarray,
    camera: Camera,
    front_camera: Camera,
    front_anchors_work: Mapping[str, np.ndarray],
    side_prior_work: Mapping[str, np.ndarray] | None,
    roi_work_xyxy: tuple[float, float, float, float] | None,
    config: NasalObservationConfig,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    contour_canvas = _external_contour_canvas(binary, name="face mask")
    contour_work = _canvas_contour_to_work(
        contour_canvas,
        camera,
        binary.shape,
        config.work_size,
    )
    named_lines, sampled_lines = _epipolar_lines_work(
        front_anchors_work,
        front_camera,
        camera,
        config,
    )
    distances = _line_distance_matrix(contour_work, sampled_lines)
    eligible = np.min(distances, axis=1) <= float(
        config.max_epipolar_distance_px
    )
    if roi_work_xyxy is not None:
        x0, y0, x1, y1 = roi_work_xyxy
        eligible &= (
            (contour_work[:, 0] >= x0)
            & (contour_work[:, 0] <= x1)
            & (contour_work[:, 1] >= y0)
            & (contour_work[:, 1] <= y1)
        )
    runs = _circular_true_runs(eligible)
    runs = [run for run in runs if len(run) >= config.min_boundary_points]
    if not runs:
        raise ValueError(
            "face mask has no continuous contour intersecting epipolar evidence"
        )

    contour_center = np.mean(contour_work, axis=0)
    curvature = _contour_curvature(contour_work)
    candidates = []
    for run in runs:
        ordered = _ordered_epipolar_candidate(
            contour_work[run],
            named_lines,
            config,
        )
        if ordered is None:
            continue
        curve, anchors, anchor_errors = ordered
        outwardness = float(
            np.mean(np.linalg.norm(curve - contour_center, axis=1))
        )
        candidate_curvature = float(
            np.percentile(curvature[run], 80.0)
        )
        prior_error = 0.0
        if side_prior_work is not None:
            prior_values = np.asarray(
                [side_prior_work[name] for name in PROFILE_PRIOR_NAMES],
                dtype=np.float64,
            )
            prior_error = float(
                np.mean(
                    np.min(
                        np.linalg.norm(
                            curve[:, None, :] - prior_values[None, :, :],
                            axis=2,
                        ),
                        axis=0,
                    )
                )
            )
        candidates.append(
            {
                "curve": curve,
                "anchors": anchors,
                "anchor_errors": anchor_errors,
                "outwardness": outwardness,
                "curvature": candidate_curvature,
                "prior_error": prior_error,
            }
        )
    if not candidates:
        raise ValueError(
            "face mask has no ordered nasal contour with epipolar support"
        )

    outward_scale = max(
        max(candidate["outwardness"] for candidate in candidates),
        1e-6,
    )
    curvature_scale = max(
        max(candidate["curvature"] for candidate in candidates),
        1e-6,
    )
    prior_scale = max(float(config.profile_prior_padding_px), 1e-6)
    for candidate in candidates:
        candidate["score"] = (
            float(np.mean(candidate["anchor_errors"]))
            / float(config.max_epipolar_distance_px)
            + 0.25 * candidate["prior_error"] / prior_scale
            - 0.20 * candidate["outwardness"] / outward_scale
            - 0.10 * candidate["curvature"] / curvature_scale
        )
    selected = min(candidates, key=lambda item: float(item["score"]))
    metadata = {
        "epipolar_lines_work": [
            np.asarray(named_lines[name], dtype=float).tolist()
            for name in PROFILE_PRIOR_NAMES
        ],
        "epipolar_anchor_names": list(PROFILE_PRIOR_NAMES),
        "max_epipolar_distance_px": float(
            config.max_epipolar_distance_px
        ),
        "candidate_count": int(len(candidates)),
        "selected_anchor_epipolar_errors_px": np.asarray(
            selected["anchor_errors"],
            dtype=float,
        ).tolist(),
        "selection_score": float(selected["score"]),
        "used_side_prior": side_prior_work is not None,
    }
    return selected["curve"], selected["anchors"], metadata


def build_profile_nasal_observation(
    image: np.ndarray,
    face_mask: np.ndarray,
    rig: ProfileRig,
    front_anchors_original: Mapping[str, Any],
    side_view: str,
    *,
    side_prior_original: Mapping[str, Any] | None = None,
    roi_original_xyxy: tuple[float, float, float, float] | None = None,
    config: NasalObservationConfig | None = None,
) -> NasalViewObservation:
    """Extract one side's continuous nasal silhouette inside a semantic ROI."""
    limits = config or NasalObservationConfig()
    if rig.reference_view != "front":
        raise ValueError("nasal observations require a front-reference rig")
    if side_view not in {"left", "right"}:
        raise ValueError("side view must be left or right")
    if "front" not in rig.cameras_by_view or side_view not in rig.cameras_by_view:
        raise ValueError(f"fixed rig is missing front or {side_view} camera")
    camera = rig.cameras_by_view[side_view]
    front_camera = rig.cameras_by_view["front"]
    frame = _validate_image(image, camera)
    semantic_view = nasal_view_for_camera(camera)
    if semantic_view not in {"subject-left", "subject-right"}:
        raise ValueError("profile nasal observation requires camera1 or camera3")
    binary = _validate_mask(face_mask, camera, name="face mask")
    if nasal_view_for_camera(front_camera) != "front":
        raise ValueError("fixed rig front camera must be camera2/front")
    front_anchors_work = _named_points_work(
        front_anchors_original,
        front_camera,
        limits,
        label="front nasal anchors",
    )
    side_prior_work = (
        None
        if side_prior_original is None
        else _named_points_work(
            side_prior_original,
            camera,
            limits,
            label="side nasal prior",
        )
    )
    roi_work = (
        None
        if roi_original_xyxy is None
        else _roi_original_to_work(
            _validate_profile_roi(roi_original_xyxy, camera),
            camera,
            limits.work_size,
        )
    )
    curve, anchors, selection_metadata = _profile_curve_from_epipolar(
        binary,
        camera,
        front_camera,
        front_anchors_work,
        side_prior_work,
        roi_work,
        limits,
    )
    variant_curves: dict[str, Mapping[str, np.ndarray]] = {
        "base": {"nasal-profile": curve},
    }
    for name, offset in (
        ("eroded", -limits.mask_perturbation_px),
        ("dilated", limits.mask_perturbation_px),
    ):
        variant = _mask_variant(binary, offset)
        try:
            candidate, _candidate_anchors, _candidate_metadata = (
                _profile_curve_from_epipolar(
                    variant,
                    camera,
                    front_camera,
                    front_anchors_work,
                    side_prior_work,
                    roi_work,
                    limits,
                )
            )
            variant_curves[name] = {"nasal-profile": candidate}
        except ValueError:
            variant_curves[name] = {}
    if roi_work is None:
        points = np.asarray(curve, dtype=np.float64)
        observation_roi = (
            float(np.min(points[:, 0])),
            float(np.min(points[:, 1])),
            float(np.max(points[:, 0]) + 1.0),
            float(np.max(points[:, 1]) + 1.0),
        )
    else:
        observation_roi = roi_work
    return _make_observation(
        semantic_view=semantic_view,
        camera=camera,
        canvas_shape=binary.shape,
        roi_work_xyxy=observation_roi,
        curves={"nasal-profile": curve},
        variant_curves=variant_curves,
        anchors=anchors,
        image=frame,
        config=limits,
        coordinate_extras=selection_metadata,
    )


def build_multiview_nasal_observations(
    images_by_view: Mapping[str, np.ndarray],
    front_semantic_nose_mask: np.ndarray,
    side_face_masks: Mapping[str, np.ndarray],
    rig: ProfileRig,
    front_anchors_original: Mapping[str, Any],
    *,
    centerline_x_original: float,
    side_priors_original: (
        Mapping[str, Mapping[str, Any]] | None
    ) = None,
    side_rois_original: (
        Mapping[str, tuple[float, float, float, float]] | None
    ) = None,
    config: NasalObservationConfig | None = None,
) -> NasalObservationBundle:
    """Build front and bilateral profile evidence from one fixed-rig capture."""
    limits = config or NasalObservationConfig()
    missing_images = [
        view for view in ("left", "front", "right")
        if view not in images_by_view
    ]
    if missing_images:
        raise ValueError(
            "nasal images are missing views: " + ", ".join(missing_images)
        )
    missing_masks = [
        view for view in ("left", "right")
        if view not in side_face_masks
    ]
    if missing_masks:
        raise ValueError(
            "side face masks are missing views: " + ", ".join(missing_masks)
        )
    front_camera = rig.cameras_by_view.get("front")
    if front_camera is None:
        raise ValueError("fixed rig is missing front camera")
    front = build_front_nasal_observation(
        images_by_view["front"],
        front_semantic_nose_mask,
        front_camera,
        centerline_x_original=centerline_x_original,
        config=limits,
    )
    profiles = {}
    for side_view in ("left", "right"):
        profiles[side_view] = build_profile_nasal_observation(
            images_by_view[side_view],
            side_face_masks[side_view],
            rig,
            front_anchors_original,
            side_view,
            side_prior_original=(
                None
                if side_priors_original is None
                else side_priors_original.get(side_view)
            ),
            roi_original_xyxy=(
                None
                if side_rois_original is None
                else side_rois_original.get(side_view)
            ),
            config=limits,
        )
    return NasalObservationBundle(
        front=front,
        subject_left=profiles["left"],
        subject_right=profiles["right"],
    )

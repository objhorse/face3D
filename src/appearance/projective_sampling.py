"""Strict camera projection and texture-sampling coordinate checks."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Union

import numpy as np


MetricValue = Union[int, float, None]
DEFAULT_POSITIVE_DEPTH_EPSILON = 1e-4
_CANONICAL_FLOAT_DTYPE = np.dtype(np.float32)


def _read_only_copy(value: np.ndarray, *, dtype: np.dtype) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        result = np.array(value, dtype=dtype, copy=True)
    result.setflags(write=False)
    return result


def _binary_mask(value: np.ndarray, shape: tuple[int, ...], name: str) -> np.ndarray:
    mask = np.asarray(value)
    if mask.shape != shape:
        raise ValueError(f"{name} must have shape {shape}")
    if np.issubdtype(mask.dtype, np.bool_):
        return mask.astype(bool, copy=False)
    if not np.issubdtype(mask.dtype, np.number) or np.issubdtype(
        mask.dtype, np.complexfloating
    ):
        raise ValueError(f"{name} must contain bool or finite 0/1 values")
    if not np.isfinite(mask).all() or not np.all((mask == 0) | (mask == 1)):
        raise ValueError(f"{name} must contain bool or finite 0/1 values")
    return mask.astype(bool, copy=False)


@dataclass(frozen=True)
class ProjectionCoordinates:
    """World points and their immutable coordinates in one camera."""

    camera_points: np.ndarray
    depth: np.ndarray
    pixel_xy: np.ndarray
    front_facing: np.ndarray

    def __post_init__(self) -> None:
        camera_points = np.asarray(self.camera_points)
        depth = np.asarray(self.depth)
        pixel_xy = np.asarray(self.pixel_xy)
        if camera_points.ndim != 2 or camera_points.shape[1] != 3:
            raise ValueError("camera_points must have shape (N, 3)")
        count = camera_points.shape[0]
        if depth.shape != (count,):
            raise ValueError("depth must have shape (N,)")
        if pixel_xy.shape != (count, 2):
            raise ValueError("pixel_xy must have shape (N, 2)")
        front_facing = _binary_mask(self.front_facing, (count,), "front_facing")
        if not np.isfinite(camera_points).all() or not np.isfinite(depth).all():
            raise ValueError("camera_points and depth must contain only finite values")
        if np.isinf(pixel_xy).any() or not np.isfinite(pixel_xy[front_facing]).all():
            raise ValueError("front-facing pixel coordinates must be finite")

        camera_points = _read_only_copy(
            camera_points, dtype=_CANONICAL_FLOAT_DTYPE
        )
        depth = _read_only_copy(depth, dtype=_CANONICAL_FLOAT_DTYPE)
        pixel_xy = _read_only_copy(pixel_xy, dtype=_CANONICAL_FLOAT_DTYPE)
        front_facing = _read_only_copy(front_facing, dtype=np.dtype(np.bool_))
        if not np.isfinite(camera_points).all() or not np.isfinite(depth).all():
            raise ValueError("camera_points and depth must remain finite as float32")
        if np.isinf(pixel_xy).any() or not np.isfinite(pixel_xy[front_facing]).all():
            raise ValueError("front-facing pixel coordinates must remain finite as float32")
        if not np.allclose(
            depth,
            camera_points[:, 2],
            rtol=1e-6,
            atol=1e-7,
        ):
            raise ValueError("depth must match camera_points[:, 2]")

        object.__setattr__(self, "camera_points", camera_points)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "pixel_xy", pixel_xy)
        object.__setattr__(self, "front_facing", front_facing)


@dataclass(frozen=True)
class ProjectionSample:
    """Image attributes sampled from one immutable set of projected pixels."""

    coordinates: ProjectionCoordinates
    image_shape: tuple[int, int]
    in_bounds: np.ndarray
    rgb: np.ndarray
    mask: Optional[np.ndarray] = None
    depth: Optional[np.ndarray] = None
    semantic: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        count = self.coordinates.pixel_xy.shape[0]
        image_shape = tuple(int(value) for value in self.image_shape)
        if len(image_shape) != 2 or min(image_shape) <= 0:
            raise ValueError("image_shape must contain positive (height, width)")

        in_bounds = _binary_mask(self.in_bounds, (count,), "in_bounds")
        rgb = np.asarray(self.rgb)
        if rgb.shape != (count, 3) or not np.isfinite(rgb).all():
            raise ValueError("rgb must have finite shape (N, 3)")

        mask = self._optional_array(self.mask, (count,), "mask", binary=True)
        depth = self._optional_array(self.depth, (count,), "depth")
        semantic = self._optional_array(self.semantic, (count,), "semantic")

        object.__setattr__(self, "image_shape", image_shape)
        object.__setattr__(
            self,
            "in_bounds",
            _read_only_copy(in_bounds, dtype=np.dtype(np.bool_)),
        )
        object.__setattr__(
            self,
            "rgb",
            _read_only_copy(rgb, dtype=_CANONICAL_FLOAT_DTYPE),
        )
        object.__setattr__(self, "mask", mask)
        object.__setattr__(self, "depth", depth)
        object.__setattr__(self, "semantic", semantic)

    @staticmethod
    def _optional_array(
        value: Optional[np.ndarray],
        shape: tuple[int, ...],
        name: str,
        *,
        binary: bool = False,
    ) -> Optional[np.ndarray]:
        if value is None:
            return None
        if binary:
            array = _binary_mask(value, shape, name)
            return _read_only_copy(array, dtype=np.dtype(np.bool_))
        array = np.asarray(value)
        if array.shape != shape or not np.isfinite(array).all():
            raise ValueError(f"{name} must have finite shape {shape}")
        return _read_only_copy(array, dtype=_CANONICAL_FLOAT_DTYPE)


def _finite_array(value: np.ndarray, shape: tuple[Optional[int], ...], name: str) -> np.ndarray:
    with np.errstate(over="ignore", invalid="ignore"):
        array = np.asarray(value, dtype=_CANONICAL_FLOAT_DTYPE)
    shape_matches = array.ndim == len(shape) and all(
        expected is None or actual == expected
        for actual, expected in zip(array.shape, shape)
    )
    if not shape_matches:
        expected = "(" + ", ".join("N" if item is None else str(item) for item in shape) + ")"
        raise ValueError(f"{name} must have shape {expected}")
    if not np.isfinite(array).all():
        raise ValueError(f"{name} must contain only finite values")
    return array


def project_points_strict(
    points: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    *,
    epsilon: float = DEFAULT_POSITIVE_DEPTH_EPSILON,
) -> ProjectionCoordinates:
    """Project world points with a world-to-camera ``R, t`` transform."""
    world_points = _finite_array(points, (None, 3), "points")
    intrinsics = _finite_array(K, (3, 3), "K")
    rotation = _finite_array(R, (3, 3), "R")
    translation = _finite_array(t, (3,), "t")
    if not np.isfinite(epsilon) or epsilon < 0.0:
        raise ValueError("epsilon must be a finite non-negative value")

    with np.errstate(over="ignore", invalid="ignore"):
        camera_points = world_points @ rotation.T + translation
    if not np.isfinite(camera_points).all():
        raise ValueError("camera projection produced non-finite camera points")

    depth = camera_points[:, 2].copy()
    front_facing = depth > float(epsilon)
    pixel_xy = np.zeros((len(world_points), 2), dtype=_CANONICAL_FLOAT_DTYPE)
    if np.any(front_facing):
        homogeneous = camera_points[front_facing] @ intrinsics.T
        denominator = homogeneous[:, 2]
        valid_denominator = np.isfinite(denominator) & (np.abs(denominator) > epsilon)
        if not np.all(valid_denominator):
            raise ValueError("camera intrinsics produced invalid perspective depth")
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            projected = homogeneous[:, :2] / denominator[:, None]
        if not np.isfinite(projected).all():
            raise ValueError("camera projection produced non-finite pixel coordinates")
        pixel_xy[front_facing] = projected

    return ProjectionCoordinates(
        camera_points=camera_points,
        depth=depth,
        pixel_xy=pixel_xy,
        front_facing=front_facing,
    )


def sample_projected_attributes(
    coordinates: ProjectionCoordinates,
    image: np.ndarray,
    *,
    mask: Optional[np.ndarray] = None,
    depth_map: Optional[np.ndarray] = None,
    semantic_map: Optional[np.ndarray] = None,
) -> ProjectionSample:
    """Sample RGB and optional raster attributes at exactly ``pixel_xy``.

    RGB uses bilinear interpolation. Discrete masks and semantic labels use the
    lower integer pixel containing the same floating-point coordinate. Depth
    uses that same integer pixel so visibility cannot silently use a different
    warped location.
    """
    rgb_image = np.asarray(image)
    if (
        rgb_image.ndim != 3
        or rgb_image.shape[2] != 3
        or not np.issubdtype(rgb_image.dtype, np.number)
        or not np.isfinite(rgb_image).all()
    ):
        raise ValueError("image must have finite numeric shape (H, W, 3)")
    height, width = rgb_image.shape[:2]
    for value, name in (
        (mask, "mask"),
        (depth_map, "depth_map"),
        (semantic_map, "semantic_map"),
    ):
        if value is not None and np.asarray(value).shape != (height, width):
            raise ValueError(f"{name} must match image height and width")

    pixels = coordinates.pixel_xy
    in_bounds = (
        coordinates.front_facing
        & (pixels[:, 0] >= 0.0)
        & (pixels[:, 0] < float(width - 1))
        & (pixels[:, 1] >= 0.0)
        & (pixels[:, 1] < float(height - 1))
    )
    count = len(pixels)
    sampled_rgb = np.zeros((count, 3), dtype=_CANONICAL_FLOAT_DTYPE)
    sampled_mask = np.zeros(count, dtype=bool) if mask is not None else None
    sampled_depth = (
        np.full(count, np.finfo(_CANONICAL_FLOAT_DTYPE).max, dtype=_CANONICAL_FLOAT_DTYPE)
        if depth_map is not None
        else None
    )
    sampled_semantic = (
        np.zeros(count, dtype=_CANONICAL_FLOAT_DTYPE)
        if semantic_map is not None
        else None
    )

    if np.any(in_bounds):
        selected = np.flatnonzero(in_bounds)
        x = pixels[selected, 0]
        y = pixels[selected, 1]
        x0 = np.floor(x).astype(np.int32)
        y0 = np.floor(y).astype(np.int32)
        x1 = x0 + 1
        y1 = y0 + 1
        wx = (x - x0).astype(_CANONICAL_FLOAT_DTYPE)[:, None]
        wy = (y - y0).astype(_CANONICAL_FLOAT_DTYPE)[:, None]
        source = rgb_image.astype(_CANONICAL_FLOAT_DTYPE, copy=False)
        sampled_rgb[selected] = (
            source[y0, x0] * (1.0 - wx) * (1.0 - wy)
            + source[y0, x1] * wx * (1.0 - wy)
            + source[y1, x0] * (1.0 - wx) * wy
            + source[y1, x1] * wx * wy
        )
        if sampled_mask is not None:
            sampled_mask[selected] = np.asarray(mask)[y0, x0] > 0
        if sampled_depth is not None:
            sampled_values = np.asarray(depth_map, dtype=np.float32)[y0, x0]
            sampled_depth[selected] = np.nan_to_num(
                sampled_values,
                nan=np.finfo(_CANONICAL_FLOAT_DTYPE).max,
                posinf=np.finfo(_CANONICAL_FLOAT_DTYPE).max,
                neginf=-np.finfo(_CANONICAL_FLOAT_DTYPE).max,
            )
        if sampled_semantic is not None:
            sampled_semantic[selected] = np.asarray(
                semantic_map, dtype=np.float32
            )[y0, x0]

    return ProjectionSample(
        coordinates=coordinates,
        image_shape=(height, width),
        in_bounds=in_bounds,
        rgb=sampled_rgb,
        mask=sampled_mask,
        depth=sampled_depth,
        semantic=sampled_semantic,
    )


def _barycentric_batch(
    points: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    c: np.ndarray,
) -> np.ndarray:
    v0 = b - a
    v1 = c - a
    v2 = points - a
    d00 = np.dot(v0, v0)
    d01 = np.dot(v0, v1)
    d11 = np.dot(v1, v1)
    denominator = d00 * d11 - d01 * d01
    if abs(float(denominator)) < 1e-12:
        return np.full((len(points), 3), -1.0, dtype=np.float32)
    d20 = v2 @ v0
    d21 = v2 @ v1
    v = (d11 * d20 - d01 * d21) / denominator
    w = (d00 * d21 - d01 * d20) / denominator
    return np.column_stack((1.0 - v - w, v, w)).astype(np.float32)


def render_camera_depth(
    vertices: np.ndarray,
    faces: np.ndarray,
    K: np.ndarray,
    R: np.ndarray,
    t: np.ndarray,
    image_shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize camera-space depth using the strict projection contract."""
    height, width = (int(image_shape[0]), int(image_shape[1]))
    if height <= 0 or width <= 0:
        raise ValueError("image_shape must contain positive (height, width)")
    vertex_array = _finite_array(vertices, (None, 3), "vertices")
    face_array = np.asarray(faces)
    if face_array.ndim != 2 or face_array.shape[1] != 3:
        raise ValueError("faces must have shape (F, 3)")
    if not np.issubdtype(face_array.dtype, np.integer):
        raise ValueError("faces must contain integer vertex indices")
    if face_array.size and (
        int(face_array.min()) < 0 or int(face_array.max()) >= len(vertex_array)
    ):
        raise ValueError("faces contain out-of-range vertex indices")

    projection = project_points_strict(vertex_array, K, R, t)
    depth_buffer = np.full((height, width), np.inf, dtype=np.float32)
    for triangle in face_array:
        triangle_depth = projection.depth[triangle]
        if np.any(triangle_depth <= DEFAULT_POSITIVE_DEPTH_EPSILON):
            continue
        pixels = projection.pixel_xy[triangle]
        x_min = max(0, int(np.floor(np.min(pixels[:, 0]))))
        x_max = min(width - 1, int(np.ceil(np.max(pixels[:, 0]))))
        y_min = max(0, int(np.floor(np.min(pixels[:, 1]))))
        y_max = min(height - 1, int(np.ceil(np.max(pixels[:, 1]))))
        if x_min > x_max or y_min > y_max:
            continue

        xs = np.arange(x_min, x_max + 1, dtype=np.float32) + 0.5
        ys = np.arange(y_min, y_max + 1, dtype=np.float32) + 0.5
        grid_x, grid_y = np.meshgrid(xs, ys)
        samples = np.stack((grid_x.ravel(), grid_y.ravel()), axis=1)
        barycentric = _barycentric_batch(
            samples,
            pixels[0],
            pixels[1],
            pixels[2],
        )
        inside = np.all(barycentric >= -1e-5, axis=1)
        if not np.any(inside):
            continue
        selected = samples[inside]
        selected_depth = (
            barycentric[inside] @ triangle_depth.astype(np.float32)
        ).astype(np.float32)
        pixel_x = selected[:, 0].astype(np.int32)
        pixel_y = selected[:, 1].astype(np.int32)
        np.minimum.at(depth_buffer, (pixel_y, pixel_x), selected_depth)
    return depth_buffer


def _selected_displacements(
    projected: np.ndarray,
    sampled: np.ndarray,
    valid_mask: Optional[np.ndarray],
) -> np.ndarray:
    projected_array = np.asarray(projected, dtype=np.float64)
    sampled_array = np.asarray(sampled, dtype=np.float64)
    if (
        projected_array.ndim != 2
        or projected_array.shape[1] != 2
        or sampled_array.shape != projected_array.shape
    ):
        raise ValueError("projected and sampled must have matching (N, 2) shapes")

    if valid_mask is None:
        mask = np.ones(projected_array.shape[0], dtype=bool)
    else:
        mask = _binary_mask(
            valid_mask,
            (projected_array.shape[0],),
            "valid mask",
        )

    selected_projected = projected_array[mask]
    selected_sampled = sampled_array[mask]
    if not (
        np.isfinite(selected_projected).all() and np.isfinite(selected_sampled).all()
    ):
        raise ValueError("valid projected and sampled coordinates must be finite")
    with np.errstate(over="ignore", invalid="ignore"):
        displacement = np.linalg.norm(
            selected_sampled - selected_projected,
            axis=1,
        )
    if not np.isfinite(displacement).all():
        raise ValueError("sampling displacement must contain only finite values")
    return displacement


def compare_sampling_coordinates(
    projected: np.ndarray,
    sampled: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    atol: float = 1e-6,
) -> dict[str, MetricValue]:
    """Report JSON-safe displacement metrics between projected and sampled pixels.

    For an empty valid selection, displacement statistics are ``None`` and all
    ratios are ``0.0``.
    """
    if not np.isfinite(atol) or atol < 0.0:
        raise ValueError("atol must be a finite non-negative value")
    displacement = _selected_displacements(projected, sampled, valid_mask)
    count = int(displacement.size)
    if count == 0:
        return {
            "count": 0,
            "mean_displacement_px": None,
            "p50_displacement_px": None,
            "p90_displacement_px": None,
            "p95_displacement_px": None,
            "max_displacement_px": None,
            "exact_coordinate_ratio": 0.0,
            "over_1px_ratio": 0.0,
            "over_5px_ratio": 0.0,
            "over_12px_ratio": 0.0,
        }

    return {
        "count": count,
        "mean_displacement_px": float(np.mean(displacement)),
        "p50_displacement_px": float(np.percentile(displacement, 50)),
        "p90_displacement_px": float(np.percentile(displacement, 90)),
        "p95_displacement_px": float(np.percentile(displacement, 95)),
        "max_displacement_px": float(np.max(displacement)),
        "exact_coordinate_ratio": float(np.mean(displacement <= atol)),
        "over_1px_ratio": float(np.mean(displacement > 1.0)),
        "over_5px_ratio": float(np.mean(displacement > 5.0)),
        "over_12px_ratio": float(np.mean(displacement > 12.0)),
    }


def assert_strict_sampling_coordinates(
    projected: np.ndarray,
    sampled: np.ndarray,
    valid_mask: Optional[np.ndarray] = None,
    atol: float = 1e-6,
) -> dict[str, MetricValue]:
    """Reject texture sampling that moves away from the projected camera ray."""
    metrics = compare_sampling_coordinates(projected, sampled, valid_mask, atol)
    if metrics["count"] and metrics["exact_coordinate_ratio"] < 1.0:
        raise ValueError(
            "strict projective sampling rejected displaced coordinates: "
            f"max displacement {metrics['max_displacement_px']:.6g}px exceeds "
            f"atol {atol:.6g}px"
        )
    return metrics

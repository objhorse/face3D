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

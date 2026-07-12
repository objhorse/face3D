"""Smooth inverse image-sampling warps for texture registration."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.interpolate import RBFInterpolator


@dataclass(frozen=True)
class SamplingWarp:
    displacement_grid: np.ndarray
    canvas_shape: tuple[int, int]
    control_residual_before_px: float
    control_residual_after_px: float
    max_displacement_px: float

    def apply(self, points: np.ndarray, image_shape: tuple[int, int]) -> np.ndarray:
        """Map model projection pixels to registered source-image sample pixels."""
        query = np.asarray(points, dtype=np.float32)
        image_h, image_w = image_shape[:2]
        canvas_h, canvas_w = self.canvas_shape
        canvas_points = query.copy()
        canvas_points[:, 0] *= canvas_w / float(image_w)
        canvas_points[:, 1] *= canvas_h / float(image_h)
        grid_h, grid_w = self.displacement_grid.shape[:2]
        gx = canvas_points[:, 0] * (grid_w - 1) / max(canvas_w - 1, 1)
        gy = canvas_points[:, 1] * (grid_h - 1) / max(canvas_h - 1, 1)
        displacement = _bilinear_vector_sample(self.displacement_grid, gx, gy)
        displacement[:, 0] *= image_w / float(canvas_w)
        displacement[:, 1] *= image_h / float(canvas_h)
        return query + displacement


def _bilinear_vector_sample(field: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    h, w = field.shape[:2]
    x0 = np.clip(np.floor(x).astype(np.int64), 0, w - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, h - 1)
    x1 = np.clip(x0 + 1, 0, w - 1)
    y1 = np.clip(y0 + 1, 0, h - 1)
    wx = (x - np.floor(x))[:, None]
    wy = (y - np.floor(y))[:, None]
    return (
        field[y0, x0] * (1 - wx) * (1 - wy)
        + field[y0, x1] * wx * (1 - wy)
        + field[y1, x0] * (1 - wx) * wy
        + field[y1, x1] * wx * wy
    )


def _robust_controls(
    model_points: np.ndarray,
    observed_points: np.ndarray,
    max_control_displacement_px: float,
) -> tuple[np.ndarray, np.ndarray]:
    model = np.asarray(model_points, dtype=np.float32)
    observed = np.asarray(observed_points, dtype=np.float32)
    if model.shape != observed.shape or model.ndim != 2 or model.shape[1] != 2:
        raise ValueError("control points must have matching (N, 2) shapes")
    displacement = observed - model
    magnitude = np.linalg.norm(displacement, axis=1)
    finite = np.isfinite(displacement).all(axis=1)
    if finite.sum() < 6:
        raise ValueError("at least six finite registration controls are required")
    median = float(np.median(magnitude[finite]))
    mad = float(np.median(np.abs(magnitude[finite] - median)))
    robust_limit = max(6.0, median + 3.5 * max(mad, 1.0))
    keep = finite & (magnitude <= min(float(max_control_displacement_px), robust_limit))
    if keep.sum() < 6:
        raise ValueError("too few registration controls remain after outlier rejection")
    return model[keep], displacement[keep]


def build_sampling_warp(
    model_points: np.ndarray,
    observed_points: np.ndarray,
    canvas_shape: tuple[int, int],
    grid_size: int = 96,
    smoothing: float = 18.0,
    max_control_displacement_px: float = 32.0,
    max_field_displacement_px: float = 28.0,
) -> SamplingWarp:
    """Fit a bounded smooth inverse warp with zero displacement at the canvas edge."""
    height, width = map(int, canvas_shape[:2])
    model, displacement = _robust_controls(
        model_points, observed_points, max_control_displacement_px
    )
    anchors = np.array(
        [
            [0, 0], [width * 0.5, 0], [width - 1, 0],
            [0, height * 0.5], [width - 1, height * 0.5],
            [0, height - 1], [width * 0.5, height - 1], [width - 1, height - 1],
        ],
        dtype=np.float32,
    )
    control_x = np.vstack((model, anchors))
    control_y = np.vstack((displacement, np.zeros_like(anchors)))
    interpolator = RBFInterpolator(
        control_x,
        control_y,
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
    )
    grid_h = int(grid_size)
    grid_w = int(round(grid_size * width / max(height, 1)))
    xs = np.linspace(0, width - 1, grid_w, dtype=np.float32)
    ys = np.linspace(0, height - 1, grid_h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    grid = interpolator(np.stack((gx.ravel(), gy.ravel()), axis=1)).reshape(grid_h, grid_w, 2)
    magnitude = np.linalg.norm(grid, axis=2)
    scale = np.minimum(1.0, float(max_field_displacement_px) / np.maximum(magnitude, 1e-6))
    grid *= scale[:, :, None]
    warp = SamplingWarp(
        grid.astype(np.float32),
        (height, width),
        float(np.linalg.norm(displacement, axis=1).mean()),
        0.0,
        float(np.linalg.norm(grid, axis=2).max()),
    )
    warped_controls = warp.apply(model, (height, width))
    after = float(np.linalg.norm(warped_controls - (model + displacement), axis=1).mean())
    return SamplingWarp(
        warp.displacement_grid,
        warp.canvas_shape,
        warp.control_residual_before_px,
        after,
        warp.max_displacement_px,
    )


def draw_registration_overlay(
    image: np.ndarray,
    model_points: np.ndarray,
    observed_points: np.ndarray,
    warp: SamplingWarp,
) -> np.ndarray:
    canvas = np.asarray(image).copy()
    corrected = warp.apply(np.asarray(model_points), canvas.shape[:2])
    for model, observed, sample in zip(model_points, observed_points, corrected):
        cv2.circle(canvas, tuple(np.round(model).astype(int)), 2, (255, 150, 20), -1)
        cv2.circle(canvas, tuple(np.round(observed).astype(int)), 2, (40, 230, 80), -1)
        cv2.line(
            canvas,
            tuple(np.round(model).astype(int)),
            tuple(np.round(sample).astype(int)),
            (40, 220, 255),
            1,
        )
    return canvas

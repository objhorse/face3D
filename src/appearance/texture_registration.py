"""Smooth inverse image-sampling warps for texture registration."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
from typing import Any

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
    displacement_p95_px: float = 0.0
    min_jacobian: float = 1.0
    diagnostics: dict[str, Any] = field(default_factory=dict)

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


@dataclass(frozen=True)
class LocalFeatureSpec:
    name: str
    model_points: np.ndarray
    observed_points: np.ndarray
    radius_x_scale: float = 0.9
    radius_y_scale: float = 1.1
    strategy: str = "rbf"
    groups: tuple[str, ...] = ()
    max_displacement_px: float | None = None


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
        float(np.percentile(np.linalg.norm(warp.displacement_grid, axis=2), 95)),
        displacement_field_metrics(warp.displacement_grid, warp.canvas_shape)["min_jacobian"],
        {"mode": "legacy_global_rbf"},
    )


def displacement_field_metrics(
    displacement_grid: np.ndarray,
    canvas_shape: tuple[int, int],
) -> dict[str, float]:
    field = np.asarray(displacement_grid, dtype=np.float64)
    if field.ndim != 3 or field.shape[2] != 2:
        raise ValueError("displacement grid must have shape (H, W, 2)")
    canvas_h, canvas_w = map(int, canvas_shape[:2])
    step_y = (canvas_h - 1) / max(field.shape[0] - 1, 1)
    step_x = (canvas_w - 1) / max(field.shape[1] - 1, 1)
    dux_dy, dux_dx = np.gradient(field[:, :, 0], step_y, step_x)
    duy_dy, duy_dx = np.gradient(field[:, :, 1], step_y, step_x)
    jacobian = (1.0 + dux_dx) * (1.0 + duy_dy) - dux_dy * duy_dx
    magnitude = np.linalg.norm(field, axis=2)
    return {
        "min_jacobian": float(np.min(jacobian)),
        "jacobian_p05": float(np.percentile(jacobian, 5)),
        "displacement_p95_px": float(np.percentile(magnitude, 95)),
        "max_displacement_px": float(np.max(magnitude)),
    }


def _bounded_similarity_matrix(
    model_points: np.ndarray,
    observed_points: np.ndarray,
    *,
    max_translation_px: float,
    max_rotation_degrees: float,
    max_scale_delta: float,
) -> tuple[np.ndarray, dict[str, float]]:
    model = np.asarray(model_points, dtype=np.float32)
    observed = np.asarray(observed_points, dtype=np.float32)
    finite = np.isfinite(model).all(axis=1) & np.isfinite(observed).all(axis=1)
    model = model[finite]
    observed = observed[finite]
    if len(model) < 3:
        raise ValueError("at least three global similarity controls are required")
    estimate, _inliers = cv2.estimateAffinePartial2D(
        model,
        observed,
        method=cv2.LMEDS,
    )
    if estimate is None:
        estimate = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    linear = np.asarray(estimate[:, :2], dtype=np.float64)
    scale = float(math.sqrt(max(np.linalg.det(linear), 1e-12)))
    angle = float(math.atan2(linear[1, 0], linear[0, 0]))
    max_angle = math.radians(float(max_rotation_degrees))
    bounded_scale = float(np.clip(scale, 1.0 - max_scale_delta, 1.0 + max_scale_delta))
    bounded_angle = float(np.clip(angle, -max_angle, max_angle))
    cosine = math.cos(bounded_angle)
    sine = math.sin(bounded_angle)
    bounded_linear = bounded_scale * np.array(
        [[cosine, -sine], [sine, cosine]], dtype=np.float64
    )
    model_center = model.mean(axis=0).astype(np.float64)
    observed_center = observed.mean(axis=0).astype(np.float64)
    translation = observed_center - bounded_linear @ model_center
    translation_norm = float(np.linalg.norm(translation))
    if translation_norm > float(max_translation_px):
        translation *= float(max_translation_px) / max(translation_norm, 1e-6)
    matrix = np.column_stack((bounded_linear, translation)).astype(np.float32)
    return matrix, {
        "estimated_scale": scale,
        "bounded_scale": bounded_scale,
        "estimated_rotation_degrees": math.degrees(angle),
        "bounded_rotation_degrees": math.degrees(bounded_angle),
        "translation_x_px": float(translation[0]),
        "translation_y_px": float(translation[1]),
    }


def _matrix_displacement_grid(
    matrix: np.ndarray,
    canvas_shape: tuple[int, int],
    grid_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = map(int, canvas_shape[:2])
    grid_h = int(grid_size)
    grid_w = int(round(grid_size * width / max(height, 1)))
    xs = np.linspace(0, width - 1, grid_w, dtype=np.float32)
    ys = np.linspace(0, height - 1, grid_h, dtype=np.float32)
    gx, gy = np.meshgrid(xs, ys)
    points = np.stack((gx, gy), axis=2)
    transformed = points @ matrix[:, :2].T + matrix[:, 2]
    return (transformed - points).astype(np.float32), gx, gy


def build_layered_feature_warp(
    global_model_points: np.ndarray,
    global_observed_points: np.ndarray,
    local_model_points: np.ndarray,
    local_observed_points: np.ndarray,
    canvas_shape: tuple[int, int],
    *,
    grid_size: int = 96,
    smoothing: float = 8.0,
    max_translation_px: float = 8.0,
    max_rotation_degrees: float = 1.0,
    max_scale_delta: float = 0.015,
    local_max_displacement_px: float = 16.0,
    min_jacobian: float = 0.35,
) -> SamplingWarp:
    """Fit a bounded global transform plus a nose-confined local sampling field."""
    height, width = map(int, canvas_shape[:2])
    matrix, similarity_report = _bounded_similarity_matrix(
        global_model_points,
        global_observed_points,
        max_translation_px=max_translation_px,
        max_rotation_degrees=max_rotation_degrees,
        max_scale_delta=max_scale_delta,
    )
    global_grid, gx, gy = _matrix_displacement_grid(matrix, (height, width), grid_size)
    model = np.asarray(local_model_points, dtype=np.float32)
    observed = np.asarray(local_observed_points, dtype=np.float32)
    finite = np.isfinite(model).all(axis=1) & np.isfinite(observed).all(axis=1)
    model = model[finite]
    observed = observed[finite]
    if len(model) < 4:
        raise ValueError("at least four local semantic controls are required")
    globally_warped = model @ matrix[:, :2].T + matrix[:, 2]
    residual = observed - globally_warped
    residual_magnitude = np.linalg.norm(residual, axis=1)
    residual_scale = np.minimum(
        1.0,
        float(local_max_displacement_px) / np.maximum(residual_magnitude, 1e-6),
    )
    residual = residual * residual_scale[:, None]

    x_span = max(float(np.ptp(model[:, 0])), 10.0)
    y_span = max(float(np.ptp(model[:, 1])), 10.0)
    center = model.mean(axis=0)
    radius_x = x_span * 0.75 + 12.0
    radius_y = y_span * 0.85 + 12.0
    anchor_angles = np.linspace(0.0, 2.0 * np.pi, 12, endpoint=False)
    anchors = np.stack(
        (
            center[0] + radius_x * 1.35 * np.cos(anchor_angles),
            center[1] + radius_y * 1.35 * np.sin(anchor_angles),
        ),
        axis=1,
    ).astype(np.float32)
    interpolator = RBFInterpolator(
        np.vstack((model, anchors)),
        np.vstack((residual, np.zeros_like(anchors))),
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
    )
    query = np.stack((gx.ravel(), gy.ravel()), axis=1)
    local_grid = interpolator(query).reshape(gx.shape[0], gx.shape[1], 2)
    radius = np.sqrt(
        ((gx - center[0]) / radius_x) ** 2
        + ((gy - center[1]) / radius_y) ** 2
    )
    influence = np.clip((1.35 - radius) / 0.45, 0.0, 1.0)
    influence = influence * influence * (3.0 - 2.0 * influence)
    local_grid *= influence[:, :, None]
    local_magnitude = np.linalg.norm(local_grid, axis=2)
    local_scale = np.minimum(
        1.0,
        float(local_max_displacement_px) / np.maximum(local_magnitude, 1e-6),
    )
    local_grid *= local_scale[:, :, None]
    local_grid = local_grid.astype(np.float32)
    full_metrics = displacement_field_metrics(global_grid + local_grid, (height, width))
    local_scale_applied = 0.0
    combined = global_grid
    metrics = displacement_field_metrics(combined, (height, width))
    for factor in np.linspace(1.0, 0.0, 21):
        candidate = global_grid + float(factor) * local_grid
        candidate_metrics = displacement_field_metrics(candidate, (height, width))
        if candidate_metrics["min_jacobian"] >= float(min_jacobian):
            combined = candidate
            metrics = candidate_metrics
            local_scale_applied = float(factor)
            break
    fallback = local_scale_applied <= 1e-6

    before = float(np.linalg.norm(observed - model, axis=1).mean())
    provisional = SamplingWarp(
        displacement_grid=combined.astype(np.float32),
        canvas_shape=(height, width),
        control_residual_before_px=before,
        control_residual_after_px=0.0,
        max_displacement_px=metrics["max_displacement_px"],
        displacement_p95_px=metrics["displacement_p95_px"],
        min_jacobian=metrics["min_jacobian"],
        diagnostics={
            "mode": "bounded_similarity_plus_local_feature",
            "local_fallback": bool(fallback),
            "local_scale_applied": local_scale_applied,
            "full_local_min_jacobian": full_metrics["min_jacobian"],
            **similarity_report,
        },
    )
    after = float(
        np.linalg.norm(provisional.apply(model, (height, width)) - observed, axis=1).mean()
    )
    return SamplingWarp(
        displacement_grid=provisional.displacement_grid,
        canvas_shape=provisional.canvas_shape,
        control_residual_before_px=before,
        control_residual_after_px=after,
        max_displacement_px=provisional.max_displacement_px,
        displacement_p95_px=provisional.displacement_p95_px,
        min_jacobian=provisional.min_jacobian,
        diagnostics=provisional.diagnostics,
    )


def _local_feature_displacement_grid(
    spec: LocalFeatureSpec,
    matrix: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    smoothing: float,
    local_max_displacement_px: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    model = np.asarray(spec.model_points, dtype=np.float32)
    observed = np.asarray(spec.observed_points, dtype=np.float32)
    if model.shape != observed.shape or model.ndim != 2 or model.shape[1] != 2:
        raise ValueError(
            f"local feature {spec.name} controls must have matching (N, 2) shapes"
        )
    finite = np.isfinite(model).all(axis=1) & np.isfinite(observed).all(axis=1)
    model = model[finite]
    observed = observed[finite]
    groups = tuple(group for group, keep in zip(spec.groups, finite) if keep)
    if len(model) < 4:
        raise ValueError(
            f"local feature {spec.name} requires at least four finite controls"
        )
    feature_max_displacement_px = (
        float(spec.max_displacement_px)
        if spec.max_displacement_px is not None
        else float(local_max_displacement_px)
    )
    if spec.strategy in {"ordered_nasal", "ordered_nasal_reference"}:
        from src.appearance.nasal_local_texture import (
            build_ordered_nasal_displacement_grid,
        )

        return build_ordered_nasal_displacement_grid(
            model,
            observed,
            groups,
            matrix,
            gx,
            gy,
            max_displacement_px=feature_max_displacement_px,
            direct_nostril_correction=(
                spec.strategy == "ordered_nasal_reference"
            ),
        )
    if spec.strategy != "rbf":
        raise ValueError(
            f"unsupported local feature strategy for {spec.name}: "
            f"{spec.strategy}"
        )
    globally_warped = model @ matrix[:, :2].T + matrix[:, 2]
    residual = observed - globally_warped
    residual_magnitude = np.linalg.norm(residual, axis=1)
    residual_scale = np.minimum(
        1.0,
        feature_max_displacement_px
        / np.maximum(residual_magnitude, 1e-6),
    )
    residual = residual * residual_scale[:, None]

    x_span = max(float(np.ptp(model[:, 0])), 4.0)
    y_span = max(float(np.ptp(model[:, 1])), 2.0)
    center = model.mean(axis=0)
    radius_x = max(x_span * float(spec.radius_x_scale) + 4.0, 8.0)
    radius_y = max(y_span * float(spec.radius_y_scale) + 6.0, 9.0)
    anchor_angles = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
    anchors = np.stack(
        (
            center[0] + radius_x * 1.15 * np.cos(anchor_angles),
            center[1] + radius_y * 1.15 * np.sin(anchor_angles),
        ),
        axis=1,
    ).astype(np.float32)
    interpolator = RBFInterpolator(
        np.vstack((model, anchors)),
        np.vstack((residual, np.zeros_like(anchors))),
        kernel="thin_plate_spline",
        smoothing=float(smoothing),
    )
    query = np.stack((gx.ravel(), gy.ravel()), axis=1)
    grid = interpolator(query).reshape(gx.shape[0], gx.shape[1], 2)
    radius = np.sqrt(
        ((gx - center[0]) / radius_x) ** 2
        + ((gy - center[1]) / radius_y) ** 2
    )
    influence = np.clip((1.18 - radius) / 0.30, 0.0, 1.0)
    influence = influence * influence * (3.0 - 2.0 * influence)
    grid *= influence[:, :, None]
    magnitude = np.linalg.norm(grid, axis=2)
    scale = np.minimum(
        1.0,
        feature_max_displacement_px / np.maximum(magnitude, 1e-6),
    )
    grid *= scale[:, :, None]
    return grid.astype(np.float32), {
        "control_count": int(len(model)),
        "center": center.astype(float).tolist(),
        "radius_x_px": float(radius_x),
        "radius_y_px": float(radius_y),
        "control_residual_before_px": float(
            np.linalg.norm(observed - globally_warped, axis=1).mean()
        ),
        "requested_max_displacement_px": float(
            residual_magnitude.max(initial=0.0)
        ),
    }


def build_multi_feature_warp(
    global_model_points: np.ndarray,
    global_observed_points: np.ndarray,
    local_features: tuple[LocalFeatureSpec, ...],
    canvas_shape: tuple[int, int],
    *,
    grid_size: int = 128,
    smoothing: float = 0.35,
    max_translation_px: float = 8.0,
    max_rotation_degrees: float = 1.0,
    max_scale_delta: float = 0.015,
    local_max_displacement_px: float = 16.0,
    min_jacobian: float = 0.35,
    independent_feature_backtracking: bool = False,
) -> SamplingWarp:
    """Compose bounded global registration with independent semantic feature fields."""
    height, width = map(int, canvas_shape[:2])
    matrix, similarity_report = _bounded_similarity_matrix(
        global_model_points,
        global_observed_points,
        max_translation_px=max_translation_px,
        max_rotation_degrees=max_rotation_degrees,
        max_scale_delta=max_scale_delta,
    )
    global_grid, gx, gy = _matrix_displacement_grid(
        matrix,
        (height, width),
        grid_size,
    )
    local_sum = np.zeros_like(global_grid, dtype=np.float32)
    feature_grids: list[tuple[str, np.ndarray]] = []
    feature_reports: dict[str, dict[str, Any]] = {}
    controls_model = []
    controls_observed = []
    for spec in local_features:
        if spec.name in feature_reports:
            raise ValueError(f"duplicate local feature name: {spec.name}")
        grid, feature_report = _local_feature_displacement_grid(
            spec,
            matrix,
            gx,
            gy,
            smoothing=float(smoothing),
            local_max_displacement_px=float(local_max_displacement_px),
        )
        local_sum += grid
        feature_grids.append((spec.name, grid))
        feature_reports[spec.name] = feature_report
        controls_model.append(np.asarray(spec.model_points, dtype=np.float32))
        controls_observed.append(
            np.asarray(spec.observed_points, dtype=np.float32)
        )

    full_metrics = displacement_field_metrics(
        global_grid + local_sum,
        (height, width),
    )
    selected_factor = 0.0
    selected_feature_factors: dict[str, float] = {}
    combined = global_grid.copy()
    metrics = displacement_field_metrics(combined, (height, width))
    if independent_feature_backtracking:
        for feature_name, feature_grid in feature_grids:
            feature_factor = 0.0
            for factor in np.linspace(1.0, 0.0, 41):
                candidate = combined + float(factor) * feature_grid
                candidate_metrics = displacement_field_metrics(
                    candidate,
                    (height, width),
                )
                if candidate_metrics["min_jacobian"] >= float(min_jacobian):
                    combined = candidate.astype(np.float32)
                    metrics = candidate_metrics
                    feature_factor = float(factor)
                    break
            selected_feature_factors[feature_name] = feature_factor
        selected_factor = (
            float(min(selected_feature_factors.values()))
            if selected_feature_factors
            else 0.0
        )
    else:
        for factor in np.linspace(1.0, 0.0, 41):
            candidate = global_grid + float(factor) * local_sum
            candidate_metrics = displacement_field_metrics(
                candidate,
                (height, width),
            )
            if candidate_metrics["min_jacobian"] >= float(min_jacobian):
                combined = candidate.astype(np.float32)
                metrics = candidate_metrics
                selected_factor = float(factor)
                break
        selected_feature_factors = {
            feature_name: selected_factor
            for feature_name, _feature_grid in feature_grids
        }

    provisional = SamplingWarp(
        displacement_grid=np.asarray(combined, dtype=np.float32),
        canvas_shape=(height, width),
        control_residual_before_px=0.0,
        control_residual_after_px=0.0,
        max_displacement_px=metrics["max_displacement_px"],
        displacement_p95_px=metrics["displacement_p95_px"],
        min_jacobian=metrics["min_jacobian"],
        diagnostics={},
    )
    if controls_model:
        model_all = np.vstack(controls_model)
        observed_all = np.vstack(controls_observed)
        globally_warped = (
            model_all @ matrix[:, :2].T + matrix[:, 2]
        )
        before = float(
            np.linalg.norm(globally_warped - observed_all, axis=1).mean()
        )
        after = float(
            np.linalg.norm(
                provisional.apply(model_all, (height, width)) - observed_all,
                axis=1,
            ).mean()
        )
    else:
        before = 0.0
        after = 0.0
    for spec in local_features:
        model = np.asarray(spec.model_points, dtype=np.float32)
        observed = np.asarray(spec.observed_points, dtype=np.float32)
        corrected = provisional.apply(model, (height, width))
        control_residuals = np.linalg.norm(corrected - observed, axis=1)
        feature_reports[spec.name]["control_residual_after_px"] = float(
            control_residuals.mean()
        )
        feature_reports[spec.name]["control_residuals_after_px"] = (
            control_residuals.astype(float).tolist()
        )
        if spec.groups and len(spec.groups) == len(control_residuals):
            group_residuals: dict[str, list[float]] = {}
            for group, residual in zip(spec.groups, control_residuals):
                group_residuals.setdefault(group, []).append(float(residual))
            feature_reports[spec.name]["group_residual_after_px"] = {
                group: {
                    "mean": float(np.mean(values)),
                    "max": float(np.max(values)),
                    "count": int(len(values)),
                }
                for group, values in group_residuals.items()
            }
    diagnostics = {
        "mode": "bounded_similarity_plus_multi_local_features",
        "features": feature_reports,
        "local_scale_applied": selected_factor,
        "feature_scale_applied": selected_feature_factors,
        "independent_feature_backtracking": bool(
            independent_feature_backtracking
        ),
        "local_fallback": bool(selected_factor <= 1e-6),
        "full_local_min_jacobian": full_metrics["min_jacobian"],
        **similarity_report,
    }
    return SamplingWarp(
        displacement_grid=provisional.displacement_grid,
        canvas_shape=provisional.canvas_shape,
        control_residual_before_px=before,
        control_residual_after_px=after,
        max_displacement_px=provisional.max_displacement_px,
        displacement_p95_px=provisional.displacement_p95_px,
        min_jacobian=provisional.min_jacobian,
        diagnostics=diagnostics,
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

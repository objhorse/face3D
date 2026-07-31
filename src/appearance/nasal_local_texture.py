"""Order-preserving nasal texture registration and UV-local compositing."""

from __future__ import annotations

from typing import Any, Sequence

import cv2
import numpy as np


def _smoothstep(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, 0.0, 1.0)
    return clipped * clipped * (3.0 - 2.0 * clipped)


def _weighted_isotonic_increasing(
    values: np.ndarray,
    weights: np.ndarray,
) -> np.ndarray:
    """Project values onto a nondecreasing sequence with weighted PAVA."""
    source = np.asarray(values, dtype=np.float64)
    source_weights = np.asarray(weights, dtype=np.float64)
    if source.ndim != 1 or source.shape != source_weights.shape:
        raise ValueError("isotonic values and weights must be matching vectors")
    if len(source) == 0 or np.any(source_weights <= 0):
        raise ValueError("isotonic regression requires positive weights")

    levels: list[float] = []
    block_weights: list[float] = []
    starts: list[int] = []
    ends: list[int] = []
    for index, (value, weight) in enumerate(zip(source, source_weights)):
        levels.append(float(value))
        block_weights.append(float(weight))
        starts.append(index)
        ends.append(index + 1)
        while len(levels) >= 2 and levels[-2] > levels[-1]:
            merged_weight = block_weights[-2] + block_weights[-1]
            merged_level = (
                levels[-2] * block_weights[-2]
                + levels[-1] * block_weights[-1]
            ) / merged_weight
            levels[-2:] = [merged_level]
            block_weights[-2:] = [merged_weight]
            ends[-2:] = [ends[-1]]
            starts.pop()

    result = np.empty_like(source)
    for level, start, end in zip(levels, starts, ends):
        result[start:end] = level
    return result


def _aggregate_horizontal_knots(
    model_x: np.ndarray,
    local_target_x: np.ndarray,
    control_weights: np.ndarray,
    *,
    merge_tolerance_px: float = 0.75,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(model_x, kind="stable")
    x = np.asarray(model_x, dtype=np.float64)[order]
    target = np.asarray(local_target_x, dtype=np.float64)[order]
    weights = np.asarray(control_weights, dtype=np.float64)[order]
    knot_x: list[float] = []
    knot_target: list[float] = []
    knot_weight: list[float] = []
    start = 0
    while start < len(x):
        end = start + 1
        while end < len(x) and x[end] - x[end - 1] <= merge_tolerance_px:
            end += 1
        weight = weights[start:end]
        weight_sum = float(weight.sum())
        knot_x.append(float(np.sum(x[start:end] * weight) / weight_sum))
        knot_target.append(
            float(np.sum(target[start:end] * weight) / weight_sum)
        )
        knot_weight.append(weight_sum)
        start = end
    return (
        np.asarray(knot_x, dtype=np.float64),
        np.asarray(knot_target, dtype=np.float64),
        np.asarray(knot_weight, dtype=np.float64),
    )


def _semantic_weights(groups: Sequence[str], count: int) -> np.ndarray:
    if not groups:
        return np.ones(count, dtype=np.float64)
    if len(groups) != count:
        raise ValueError("nasal semantic groups must match the controls")
    values = {
        "tip_anchor": 0.8,
        "lower_left": 1.5,
        "lower_right": 1.5,
        "nostril_left": 2.2,
        "nostril_right": 2.2,
        "named_landmarks": 1.0,
    }
    return np.asarray([values.get(group, 1.0) for group in groups], dtype=np.float64)


def build_ordered_nasal_displacement_grid(
    model_points: np.ndarray,
    observed_points: np.ndarray,
    groups: Sequence[str],
    matrix: np.ndarray,
    gx: np.ndarray,
    gy: np.ndarray,
    *,
    max_displacement_px: float = 28.0,
    minimum_horizontal_slope: float = 0.40,
    direct_nostril_correction: bool = False,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Build a nasal field whose horizontal sampling order cannot reverse.

    The horizontal component is a monotone one-dimensional map blended by a
    y-only envelope. The vertical component is a smooth semantic-control blend.
    """
    model = np.asarray(model_points, dtype=np.float32)
    observed = np.asarray(observed_points, dtype=np.float32)
    if model.shape != observed.shape or model.ndim != 2 or model.shape[1] != 2:
        raise ValueError("nasal controls must have matching (N, 2) shapes")
    if len(model) < 4:
        raise ValueError("ordered nasal registration requires four controls")
    if np.asarray(gx).shape != np.asarray(gy).shape:
        raise ValueError("sampling grid coordinates must have matching shapes")

    finite = np.isfinite(model).all(axis=1) & np.isfinite(observed).all(axis=1)
    model = model[finite]
    observed = observed[finite]
    finite_groups = tuple(group for group, keep in zip(groups, finite) if keep)
    if len(model) < 4:
        raise ValueError("too few finite nasal controls")
    weights = _semantic_weights(finite_groups, len(model))

    globally_warped = model @ matrix[:, :2].T + matrix[:, 2]
    residual = observed - globally_warped
    magnitude = np.linalg.norm(residual, axis=1)
    scale = np.minimum(
        1.0,
        float(max_displacement_px) / np.maximum(magnitude, 1e-6),
    )
    residual = residual * scale[:, None]

    x_span = max(float(np.ptp(model[:, 0])), 8.0)
    y_span = max(float(np.ptp(model[:, 1])), 4.0)
    margin_x = max(10.0, 0.32 * x_span)
    outer_left = float(model[:, 0].min() - margin_x)
    outer_right = float(model[:, 0].max() + margin_x)
    local_target_x = model[:, 0].astype(np.float64) + residual[:, 0]
    knot_x, knot_target, knot_weights = _aggregate_horizontal_knots(
        model[:, 0],
        local_target_x,
        weights,
    )
    knot_x = np.r_[outer_left, knot_x, outer_right]
    knot_target = np.r_[outer_left, knot_target, outer_right]
    knot_weights = np.r_[1.0e6, knot_weights, 1.0e6]

    minimum_slope = float(np.clip(minimum_horizontal_slope, 0.02, 0.95))
    transformed_target = knot_target - minimum_slope * knot_x
    transformed_target = _weighted_isotonic_increasing(
        transformed_target,
        knot_weights,
    )
    monotone_target = transformed_target + minimum_slope * knot_x
    monotone_target[0] = outer_left
    monotone_target[-1] = outer_right
    for index in range(1, len(monotone_target)):
        minimum = monotone_target[index - 1] + minimum_slope * (
            knot_x[index] - knot_x[index - 1]
        )
        monotone_target[index] = max(monotone_target[index], minimum)
    monotone_target[-1] = outer_right
    for index in range(len(monotone_target) - 2, -1, -1):
        maximum = monotone_target[index + 1] - minimum_slope * (
            knot_x[index + 1] - knot_x[index]
        )
        monotone_target[index] = min(monotone_target[index], maximum)
    monotone_target[0] = outer_left

    mapped_x = np.interp(
        np.asarray(gx, dtype=np.float64),
        knot_x,
        monotone_target,
    )
    horizontal = mapped_x - np.asarray(gx, dtype=np.float64)
    horizontal[
        (np.asarray(gx) <= outer_left) | (np.asarray(gx) >= outer_right)
    ] = 0.0

    inner_y_min = float(model[:, 1].min() - 2.0)
    inner_y_max = float(model[:, 1].max() + 2.0)
    margin_y = max(
        18.0,
        1.10 * y_span,
        2.5 * float(np.max(np.abs(residual[:, 1]), initial=0.0)),
    )
    y = np.asarray(gy, dtype=np.float64)
    vertical_envelope = np.ones_like(y)
    above = y < inner_y_min
    below = y > inner_y_max
    vertical_envelope[above] = _smoothstep(
        (y[above] - (inner_y_min - margin_y)) / margin_y
    )
    vertical_envelope[below] = _smoothstep(
        ((inner_y_max + margin_y) - y[below]) / margin_y
    )
    vertical_envelope[
        (y <= inner_y_min - margin_y) | (y >= inner_y_max + margin_y)
    ] = 0.0
    horizontal *= vertical_envelope

    sigma_x = max(7.0, 0.15 * x_span)
    gaussian_sum = np.zeros_like(y)
    vertical_sum = np.zeros_like(y)
    for point, delta, weight in zip(model, residual, weights):
        gaussian = float(weight) * np.exp(
            -0.5 * ((np.asarray(gx) - float(point[0])) / sigma_x) ** 2
        )
        gaussian_sum += gaussian
        vertical_sum += gaussian * float(delta[1])
    vertical = vertical_sum / np.maximum(gaussian_sum, 1e-8)
    x_support = _smoothstep(
        np.minimum(
            (np.asarray(gx) - outer_left) / max(margin_x, 1.0),
            (outer_right - np.asarray(gx)) / max(margin_x, 1.0),
        )
    )
    vertical *= vertical_envelope * x_support
    vertical = np.clip(
        vertical,
        -float(max_displacement_px),
        float(max_displacement_px),
    )

    grid = np.stack((horizontal, vertical), axis=2).astype(np.float32)
    direct_corrections: list[dict[str, Any]] = []
    if direct_nostril_correction:
        grid_h, grid_w = grid.shape[:2]
        canvas_x_max = float(np.max(gx))
        canvas_y_max = float(np.max(gy))
        for index, group in enumerate(finite_groups):
            if group not in {"nostril_left", "nostril_right"}:
                continue
            point = model[index]
            sample_x = float(point[0]) * (grid_w - 1) / max(
                canvas_x_max,
                1.0,
            )
            sample_y = float(point[1]) * (grid_h - 1) / max(
                canvas_y_max,
                1.0,
            )
            x0 = int(np.clip(np.floor(sample_x), 0, grid_w - 1))
            y0 = int(np.clip(np.floor(sample_y), 0, grid_h - 1))
            x1 = min(x0 + 1, grid_w - 1)
            y1 = min(y0 + 1, grid_h - 1)
            wx = sample_x - np.floor(sample_x)
            wy = sample_y - np.floor(sample_y)
            sampled = (
                grid[y0, x0] * (1.0 - wx) * (1.0 - wy)
                + grid[y0, x1] * wx * (1.0 - wy)
                + grid[y1, x0] * (1.0 - wx) * wy
                + grid[y1, x1] * wx * wy
            )
            current = globally_warped[index] + sampled
            remaining = observed[index] - current
            radius_x = max(34.0, 3.5 * abs(float(remaining[0])) + 14.0)
            radius_y = max(24.0, 3.5 * abs(float(remaining[1])) + 12.0)
            radius = np.sqrt(
                ((np.asarray(gx) - float(point[0])) / radius_x) ** 2
                + ((np.asarray(gy) - float(point[1])) / radius_y) ** 2
            )
            bump = _smoothstep(1.0 - radius)
            grid += bump[:, :, None].astype(np.float32) * remaining[
                None,
                None,
                :,
            ]
            direct_corrections.append(
                {
                    "group": group,
                    "remaining_before_px": float(np.linalg.norm(remaining)),
                    "radius_x_px": float(radius_x),
                    "radius_y_px": float(radius_y),
                }
            )
    requested_target = model[:, 0].astype(np.float64) + residual[:, 0]
    fitted_target = np.interp(model[:, 0], knot_x, monotone_target)
    return grid, {
        "strategy": "ordered_nasal",
        "control_count": int(len(model)),
        "requested_max_displacement_px": float(magnitude.max(initial=0.0)),
        "clipped_control_count": int(np.count_nonzero(scale < 1.0)),
        "horizontal_knot_count": int(len(knot_x)),
        "horizontal_knot_x": knot_x.astype(float).tolist(),
        "horizontal_target_x": monotone_target.astype(float).tolist(),
        "minimum_horizontal_slope": minimum_slope,
        "horizontal_projection_adjustment_mean_px": float(
            np.abs(fitted_target - requested_target).mean()
        ),
        "radius_x_px": float((outer_right - outer_left) * 0.5),
        "radius_y_px": float((inner_y_max - inner_y_min) * 0.5 + margin_y),
        "center": [
            float((outer_left + outer_right) * 0.5),
            float((inner_y_min + inner_y_max) * 0.5),
        ],
        "control_residual_before_px": float(
            np.linalg.norm(observed - globally_warped, axis=1).mean()
        ),
        "direct_nostril_correction": bool(direct_nostril_correction),
        "direct_nostril_corrections": direct_corrections,
    }


def build_nasal_uv_alpha(
    vertices: np.ndarray,
    faces: np.ndarray,
    uv_vertices: np.ndarray,
    uv_faces: np.ndarray,
    camera: dict[str, np.ndarray],
    nose_mask: np.ndarray,
    *,
    texture_size: int,
    source_mask_dilate_px: int = 3,
    feather_px: float = 5.0,
    depth_tolerance_ratio: float = 0.01,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Project the visible front nasal mask into UV space."""
    from src.appearance.projective_sampling import render_camera_depth
    from src.module3_texture import rasterize_uv_map

    mesh_vertices = np.asarray(vertices, dtype=np.float32)
    mesh_faces = np.asarray(faces, dtype=np.int64)
    tri_map, bary_map = rasterize_uv_map(
        np.asarray(uv_vertices, dtype=np.float32),
        np.asarray(uv_faces, dtype=np.int64),
        int(texture_size),
    )
    image_mask = (np.asarray(nose_mask) > 0).astype(np.uint8)
    if image_mask.ndim != 2:
        raise ValueError("nose mask must be two-dimensional")
    if source_mask_dilate_px > 0:
        kernel_size = int(source_mask_dilate_px) * 2 + 1
        image_mask = cv2.dilate(
            image_mask,
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (kernel_size, kernel_size),
            ),
        )

    valid_y, valid_x = np.where(tri_map >= 0)
    support = np.zeros_like(tri_map, dtype=np.uint8)
    if len(valid_y) == 0:
        return support.astype(np.float32), {
            "owned_texels": 0,
            "uv_texels": 0,
            "coverage_fraction": 0.0,
        }
    face_indices = tri_map[valid_y, valid_x]
    barycentric = bary_map[valid_y, valid_x]
    positions = (
        mesh_vertices[mesh_faces[face_indices]]
        * barycentric[:, :, None]
    ).sum(axis=1)
    rotation = np.asarray(camera["R"], dtype=np.float64)
    translation = np.asarray(camera["t"], dtype=np.float64).reshape(3)
    intrinsics = np.asarray(camera["K"], dtype=np.float64)
    camera_points = positions @ rotation.T + translation
    homogeneous = camera_points @ intrinsics.T
    projected = homogeneous[:, :2] / np.maximum(
        homogeneous[:, 2:3],
        1e-8,
    )
    px = np.round(projected[:, 0]).astype(np.int64)
    py = np.round(projected[:, 1]).astype(np.int64)
    image_h, image_w = image_mask.shape
    inside = (
        (camera_points[:, 2] > 0)
        & (px >= 0)
        & (px < image_w)
        & (py >= 0)
        & (py < image_h)
    )

    depth = render_camera_depth(
        mesh_vertices,
        mesh_faces,
        intrinsics,
        rotation,
        translation,
        (image_h, image_w),
    )
    selected = np.zeros(len(valid_y), dtype=bool)
    inside_indices = np.flatnonzero(inside)
    if len(inside_indices):
        sample_x = px[inside_indices]
        sample_y = py[inside_indices]
        visible_depth = depth[sample_y, sample_x]
        point_depth = camera_points[inside_indices, 2]
        tolerance = np.maximum(
            1e-4,
            np.abs(point_depth) * float(depth_tolerance_ratio),
        )
        visible = (
            np.isfinite(visible_depth)
            & (np.abs(point_depth - visible_depth) <= tolerance)
        )
        semantic = image_mask[sample_y, sample_x] > 0
        selected[inside_indices] = visible & semantic
    support[valid_y[selected], valid_x[selected]] = 1
    support = cv2.morphologyEx(
        support,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)),
    )
    distance = cv2.distanceTransform(support, cv2.DIST_L2, 3)
    alpha = np.clip(distance / max(float(feather_px), 1.0), 0.0, 1.0)
    alpha[support == 0] = 0.0
    owned = int(np.count_nonzero(alpha > 0))
    uv_texels = int(np.count_nonzero(tri_map >= 0))
    return alpha.astype(np.float32), {
        "owned_texels": owned,
        "uv_texels": uv_texels,
        "coverage_fraction": float(owned / max(uv_texels, 1)),
        "source_mask_pixels": int(np.count_nonzero(image_mask)),
        "feather_px": float(feather_px),
        "source_mask_dilate_px": int(source_mask_dilate_px),
    }


def composite_pixel_locked_texture(
    baseline: np.ndarray,
    candidate: np.ndarray,
    alpha: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Blend owned UV pixels while preserving all other bytes exactly."""
    base = np.asarray(baseline)
    new = np.asarray(candidate)
    weights = np.asarray(alpha, dtype=np.float32)
    if base.shape != new.shape or base.ndim != 3 or base.shape[2] != 3:
        raise ValueError("baseline and candidate textures must match RGB shape")
    if weights.shape != base.shape[:2]:
        raise ValueError("UV alpha shape must match the texture")
    weights = np.clip(weights, 0.0, 1.0)
    owned = weights > 0.0
    result = base.copy()
    blended = np.rint(
        base.astype(np.float32) * (1.0 - weights[:, :, None])
        + new.astype(np.float32) * weights[:, :, None]
    )
    result[owned] = np.clip(blended[owned], 0, 255).astype(base.dtype)
    outside_equal = bool(np.array_equal(result[~owned], base[~owned]))
    if not outside_equal:
        raise RuntimeError("pixel lock failed outside the nasal UV ownership mask")
    changed = np.any(result != base, axis=2)
    return result, {
        "owned_texels": int(np.count_nonzero(owned)),
        "changed_texels": int(np.count_nonzero(changed)),
        "outside_exact": outside_equal,
        "maximum_channel_delta": int(
            np.max(
                np.abs(
                    result.astype(np.int16) - base.astype(np.int16)
                )
            )
        ),
    }

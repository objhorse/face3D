"""Auditable profile-depth metrics for fixed-topology face meshes."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np


REQUIRED_PROFILE_REGIONS = ("nose_tip", "mouth", "chin")


def _vertices_array(vertices: Any) -> np.ndarray:
    array = np.asarray(vertices, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3 or len(array) == 0:
        raise ValueError("vertices must have shape (N, 3)")
    if not np.isfinite(array).all():
        raise ValueError("vertices must contain only finite values")
    return array


def _region_indices(
    regions: Mapping[str, Sequence[int]],
    name: str,
    vertex_count: int,
) -> np.ndarray:
    if name not in regions:
        raise ValueError(f"required profile region is missing: {name}")
    indices = np.asarray(regions[name], dtype=np.int64).reshape(-1)
    if len(indices) == 0:
        raise ValueError(f"profile region is empty: {name}")
    if int(indices.min()) < 0 or int(indices.max()) >= vertex_count:
        raise ValueError(
            f"profile region {name} references vertices outside [0, {vertex_count})"
        )
    return np.unique(indices)


def _validate_axis(axis: int) -> int:
    axis = int(axis)
    if axis not in (0, 1, 2):
        raise ValueError("forward_axis must be 0, 1, or 2")
    return axis


def _depth_stats(values: np.ndarray, unit_scale: float) -> dict[str, float | int]:
    scaled = np.asarray(values, dtype=np.float64) * float(unit_scale)
    return {
        "count": int(len(scaled)),
        "minimum": float(np.min(scaled)),
        "mean": float(np.mean(scaled)),
        "median": float(np.median(scaled)),
        "p95": float(np.percentile(scaled, 95.0)),
        "foremost": float(np.max(scaled)),
    }


def profile_depth_metrics(
    vertices: Any,
    regions: Mapping[str, Sequence[int]],
    *,
    forward_axis: int = 2,
    forward_sign: float = 1.0,
    unit_scale: float = 1000.0,
    face_width: float | None = None,
) -> dict[str, Any]:
    """Measure relative nose, mouth, and chin depth without changing geometry."""
    verts = _vertices_array(vertices)
    axis = _validate_axis(forward_axis)
    sign = float(forward_sign)
    if not np.isfinite(sign) or sign == 0.0:
        raise ValueError("forward_sign must be finite and non-zero")
    if not np.isfinite(unit_scale) or float(unit_scale) <= 0.0:
        raise ValueError("unit_scale must be finite and positive")

    region_names = list(REQUIRED_PROFILE_REGIONS)
    for optional_name in ("subnasale", "upper_lip", "lower_lip", "mouth_center"):
        if optional_name in regions:
            region_names.append(optional_name)

    raw_depths: dict[str, np.ndarray] = {}
    region_stats: dict[str, dict[str, float | int]] = {}
    for name in region_names:
        indices = _region_indices(regions, name, len(verts))
        values = verts[indices, axis] * sign
        raw_depths[name] = values
        region_stats[name] = _depth_stats(values, unit_scale)

    width = float(np.ptp(verts[:, 0])) if face_width is None else float(face_width)
    if not np.isfinite(width) or width <= 1e-12:
        raise ValueError("face_width must be finite and positive")

    nose_raw = float(np.max(raw_depths["nose_tip"]))
    mouth_raw = float(np.max(raw_depths["mouth"]))
    chin_raw = float(np.max(raw_depths["chin"]))
    nose_to_mouth_raw = nose_raw - mouth_raw
    mouth_to_chin_raw = mouth_raw - chin_raw

    report: dict[str, Any] = {
        "vertex_count": int(len(verts)),
        "forward_axis": axis,
        "forward_sign": sign,
        "unit_scale": float(unit_scale),
        "face_width": width * float(unit_scale),
        "regions": region_stats,
        "nose_tip_minus_mouth": nose_to_mouth_raw * float(unit_scale),
        "mouth_minus_chin": mouth_to_chin_raw * float(unit_scale),
        "nose_tip_minus_mouth_face_width_ratio": nose_to_mouth_raw / width,
        "mouth_minus_chin_face_width_ratio": mouth_to_chin_raw / width,
    }

    for lip_name in ("upper_lip", "lower_lip", "mouth_center"):
        if lip_name not in raw_depths:
            continue
        delta_raw = nose_raw - float(np.max(raw_depths[lip_name]))
        report[f"nose_tip_minus_{lip_name}"] = delta_raw * float(unit_scale)
        report[f"nose_tip_minus_{lip_name}_face_width_ratio"] = delta_raw / width
    if "subnasale" in raw_depths:
        subnasale_raw = float(np.max(raw_depths["subnasale"]))
        delta_raw = subnasale_raw - mouth_raw
        report["subnasale_minus_mouth"] = delta_raw * float(unit_scale)
        report["subnasale_minus_mouth_face_width_ratio"] = delta_raw / width
    return report


def _transition_metrics(
    baseline_vertices: np.ndarray,
    candidate_vertices: np.ndarray,
    regions: Mapping[str, Sequence[int]],
    *,
    forward_axis: int,
    forward_sign: float,
    unit_scale: float,
) -> dict[str, Any]:
    if baseline_vertices.shape != candidate_vertices.shape:
        raise ValueError("profile depth comparison requires matching topology")
    mouth_indices = _region_indices(regions, "mouth", len(baseline_vertices))
    nose_indices = _region_indices(regions, "nose_tip", len(baseline_vertices))
    chin_indices = _region_indices(regions, "chin", len(baseline_vertices))
    axis = _validate_axis(forward_axis)
    signed_delta = (
        candidate_vertices[:, axis] - baseline_vertices[:, axis]
    ) * float(forward_sign) * float(unit_scale)

    baseline_metrics = profile_depth_metrics(
        baseline_vertices,
        regions,
        forward_axis=axis,
        forward_sign=forward_sign,
        unit_scale=unit_scale,
    )
    candidate_metrics = profile_depth_metrics(
        candidate_vertices,
        regions,
        forward_axis=axis,
        forward_sign=forward_sign,
        unit_scale=unit_scale,
    )
    mouth_delta = signed_delta[mouth_indices]
    return {
        "mouth_forward_minimum": float(np.min(mouth_delta)),
        "mouth_forward_mean": float(np.mean(mouth_delta)),
        "mouth_forward_median": float(np.median(mouth_delta)),
        "mouth_forward_p95": float(np.percentile(mouth_delta, 95.0)),
        "mouth_forward_maximum": float(np.max(mouth_delta)),
        "nose_forward_mean": float(np.mean(signed_delta[nose_indices])),
        "chin_forward_mean": float(np.mean(signed_delta[chin_indices])),
        "nose_lead_change": float(
            candidate_metrics["nose_tip_minus_mouth"]
            - baseline_metrics["nose_tip_minus_mouth"]
        ),
    }


def compare_expression_depth(
    neutral_vertices: Any,
    expression_vertices: Any,
    regions: Mapping[str, Sequence[int]],
    *,
    forward_axis: int = 2,
    forward_sign: float = 1.0,
    unit_scale: float = 1000.0,
    max_mean_forward: float | None = 0.5,
) -> dict[str, Any]:
    """Gate expression-induced mouth depth while retaining full diagnostics."""
    neutral = _vertices_array(neutral_vertices)
    expression = _vertices_array(expression_vertices)
    transition = _transition_metrics(
        neutral,
        expression,
        regions,
        forward_axis=forward_axis,
        forward_sign=forward_sign,
        unit_scale=unit_scale,
    )
    issues: list[str] = []
    threshold = None if max_mean_forward is None else float(max_mean_forward)
    if threshold is not None:
        if not np.isfinite(threshold) or threshold < 0.0:
            raise ValueError("max_mean_forward must be finite and non-negative")
        if transition["mouth_forward_mean"] > threshold:
            issues.append("mouth_forward_displacement_exceeded")
    return {
        "passed": not issues,
        "issues": issues,
        "max_mean_forward": threshold,
        **transition,
        "neutral": profile_depth_metrics(
            neutral,
            regions,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            unit_scale=unit_scale,
        ),
        "expression": profile_depth_metrics(
            expression,
            regions,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            unit_scale=unit_scale,
        ),
    }


def build_profile_depth_stage_report(
    stages: Mapping[str, Any],
    regions: Mapping[str, Sequence[int]],
    *,
    forward_axis: int = 2,
    forward_sign: float = 1.0,
    unit_scale: float = 1000.0,
) -> dict[str, Any]:
    """Summarize ordered reconstruction stages and adjacent depth changes."""
    if not stages:
        raise ValueError("at least one profile depth stage is required")
    arrays = {name: _vertices_array(vertices) for name, vertices in stages.items()}
    names = list(arrays)
    stage_metrics = {
        name: profile_depth_metrics(
            arrays[name],
            regions,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            unit_scale=unit_scale,
        )
        for name in names
    }
    transitions = {}
    for previous, current in zip(names, names[1:]):
        transitions[f"{previous}_to_{current}"] = _transition_metrics(
            arrays[previous],
            arrays[current],
            regions,
            forward_axis=forward_axis,
            forward_sign=forward_sign,
            unit_scale=unit_scale,
        )
    return {
        "forward_axis": int(forward_axis),
        "forward_sign": float(forward_sign),
        "unit_scale": float(unit_scale),
        "stages": stage_metrics,
        "transitions": transitions,
    }

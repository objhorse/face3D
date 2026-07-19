from __future__ import annotations

import numpy as np
import pytest

from src.geometry.profile_depth_quality import (
    build_profile_depth_stage_report,
    compare_expression_depth,
    profile_depth_metrics,
)


def _profile_vertices() -> np.ndarray:
    return np.array(
        [
            [-1.0, 1.0, 12.0],
            [1.0, 1.0, 13.0],
            [-0.5, 0.0, 6.0],
            [0.5, 0.0, 7.0],
            [-0.5, -1.0, 3.0],
            [0.5, -1.0, 4.0],
        ],
        dtype=np.float64,
    )


def _regions() -> dict[str, list[int]]:
    return {
        "nose_tip": [0, 1],
        "mouth": [2, 3],
        "chin": [4, 5],
    }


def test_profile_relative_depth_is_translation_invariant():
    vertices = _profile_vertices()
    baseline = profile_depth_metrics(vertices, _regions(), unit_scale=1.0)
    translated = vertices + np.array([40.0, -20.0, 75.0])

    shifted = profile_depth_metrics(translated, _regions(), unit_scale=1.0)

    assert shifted["nose_tip_minus_mouth"] == pytest.approx(
        baseline["nose_tip_minus_mouth"]
    )
    assert shifted["mouth_minus_chin"] == pytest.approx(
        baseline["mouth_minus_chin"]
    )
    assert shifted["nose_tip_minus_mouth_face_width_ratio"] == pytest.approx(
        baseline["nose_tip_minus_mouth_face_width_ratio"]
    )


def test_expression_forward_mouth_displacement_is_detected():
    neutral = _profile_vertices()
    expression = neutral.copy()
    expression[_regions()["mouth"], 2] += 3.0

    comparison = compare_expression_depth(
        neutral,
        expression,
        _regions(),
        unit_scale=1.0,
        max_mean_forward=0.5,
    )

    assert comparison["passed"] is False
    assert comparison["mouth_forward_mean"] == pytest.approx(3.0)
    assert comparison["mouth_forward_p95"] == pytest.approx(3.0)
    assert comparison["nose_lead_change"] == pytest.approx(-3.0)
    assert "mouth_forward_displacement_exceeded" in comparison["issues"]


def test_invalid_vertices_and_regions_fail_explicitly():
    bad_vertices = _profile_vertices()
    bad_vertices[0, 2] = np.nan
    with pytest.raises(ValueError, match="finite"):
        profile_depth_metrics(bad_vertices, _regions())

    bad_regions = _regions()
    bad_regions["mouth"] = [99]
    with pytest.raises(ValueError, match="mouth"):
        profile_depth_metrics(_profile_vertices(), bad_regions)

    missing_regions = {"nose_tip": [0], "mouth": [1]}
    with pytest.raises(ValueError, match="chin"):
        profile_depth_metrics(_profile_vertices(), missing_regions)


def test_expression_comparison_requires_identical_topology():
    neutral = _profile_vertices()
    with pytest.raises(ValueError, match="matching topology"):
        compare_expression_depth(
            neutral,
            neutral[:-1],
            _regions(),
        )


def test_stage_report_preserves_stage_order_and_transitions():
    mean = _profile_vertices()
    mica = mean.copy()
    mica[_regions()["mouth"], 2] += 1.0
    final = mica.copy()
    final[_regions()["mouth"], 2] += 2.0

    report = build_profile_depth_stage_report(
        {"mean": mean, "mica": mica, "final": final},
        _regions(),
        unit_scale=1.0,
    )

    assert list(report["stages"]) == ["mean", "mica", "final"]
    assert report["transitions"]["mean_to_mica"]["mouth_forward_mean"] == pytest.approx(1.0)
    assert report["transitions"]["mica_to_final"]["mouth_forward_mean"] == pytest.approx(2.0)

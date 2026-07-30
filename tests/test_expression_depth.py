from __future__ import annotations

import numpy as np
import pytest

from src.geometry.expression_depth import (
    ExpressionDepthThresholds,
    constrain_expression_depth,
    constrain_expression_mouth_depth_protected,
    expression_depth_diagnostics,
    expression_regions_from_landmarks,
)


def _regions():
    return {
        "nose": np.array([0, 1]),
        "mouth": np.array([2, 3, 4]),
        "chin": np.array([5, 6]),
        "eyes": np.array([7, 8]),
    }


def test_expression_depth_detects_forward_mouth_with_closed_gap_unobserved():
    basis = np.zeros((9, 3, 2), dtype=np.float64)
    basis[[2, 3, 4], 2, 0] = 0.002
    report = expression_depth_diagnostics(basis, np.array([2.0, 0.0]), _regions())

    assert report["passed"] is False
    assert report["regions"]["mouth"]["forward_mean_mm"] == pytest.approx(4.0)
    assert "mouth_forward_mean_exceeded" in report["issues"]


def test_projection_removes_mouth_depth_without_erasing_independent_eye_mode():
    basis = np.zeros((9, 3, 2), dtype=np.float64)
    basis[[2, 3, 4], 2, 0] = 0.002
    basis[[7, 8], 1, 1] = 0.003
    original = np.array([2.0, 3.0])

    result = constrain_expression_depth(basis, original, _regions())

    assert result["selected"]["passed"] is True
    assert result["parameters"][1] == pytest.approx(original[1], abs=1e-8)
    assert result["parameters"][0] < original[0]
    assert result["selected"]["regions"]["mouth"]["forward_mean_mm"] <= 0.5 + 1e-4


def test_projection_blocks_local_lip_spike_even_when_mean_is_small():
    basis = np.zeros((9, 3, 1), dtype=np.float64)
    basis[2, 2, 0] = 0.006
    basis[3, 2, 0] = -0.002
    basis[4, 2, 0] = -0.002

    result = constrain_expression_depth(basis, np.array([1.0]), _regions())

    assert result["selected"]["passed"] is True
    assert result["selected"]["regions"]["mouth"]["forward_p95_mm"] <= 1.5 + 1e-4


def test_projection_constrains_opposing_chin_depth_without_mean_cancellation():
    basis = np.zeros((9, 3, 1), dtype=np.float64)
    basis[5, 2, 0] = 0.003
    basis[6, 2, 0] = -0.003

    result = constrain_expression_depth(basis, np.array([1.0]), _regions())

    assert result["selected"]["passed"] is True
    assert result["selected"]["regions"]["chin"]["absolute_mean_mm"] <= 0.5 + 1e-4
    assert result["selected"].get("fallback") is None


def test_expression_regions_are_derived_from_fixed_landmark_triangles():
    triangles = np.arange(68 * 3, dtype=np.int64).reshape(68, 3)
    regions = expression_regions_from_landmarks(triangles, 68 * 3)

    assert set(triangles[30]).issubset(set(regions["nose"]))
    assert set(triangles[51]).issubset(set(regions["mouth"]))
    assert set(triangles[8]).issubset(set(regions["chin"]))
    assert set(triangles[40]).issubset(set(regions["eyes"]))


def test_non_finite_expression_is_rejected():
    basis = np.zeros((9, 3, 1), dtype=np.float64)
    with pytest.raises(ValueError, match="non-finite"):
        constrain_expression_depth(basis, np.array([np.nan]), _regions())


def test_protected_constraint_prefers_mouth_mode_that_preserves_eyes():
    basis = np.zeros((9, 3, 2), dtype=np.float64)
    basis[[2, 3, 4], 2, 0] = 0.001
    basis[[2, 3, 4], 2, 1] = 0.001
    basis[[7, 8], 0, 0] = 0.004
    original = np.array([1.0, 1.0])

    result = constrain_expression_mouth_depth_protected(
        basis,
        original,
        _regions(),
    )

    assert result["selected"]["passed"] is True
    assert result["parameters"][0] == pytest.approx(original[0], abs=0.03)
    assert result["parameters"][1] < 0.0
    assert (
        result["protected_geometry_delta"]["eyes"]["xyz_mean_mm"]
        < 0.15
    )
    assert (
        result["selected"]["regions"]["mouth"]["forward_mean_mm"]
        <= 0.5 + 1e-4
    )

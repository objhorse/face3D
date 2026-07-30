from __future__ import annotations

import numpy as np
import pytest

from src.cross_view_geometry import Camera
from src.geometry.semantic_epipolar_refinement import (
    SemanticRefinementThresholds,
    estimate_local_depth_surface,
    refine_semantic_pair,
)


def _camera(name: str, view: str, center_x: float) -> Camera:
    rotation = np.eye(3, dtype=np.float64)
    center = np.array([center_x, 0.0, 0.0], dtype=np.float64)
    return Camera(
        name=name,
        view=view,
        image_size=(640, 480),
        K=np.array(
            ((700.0, 0.0, 320.0), (0.0, 700.0, 240.0), (0.0, 0.0, 1.0)),
            dtype=np.float64,
        ),
        dist=np.zeros(5, dtype=np.float64),
        R_rig_to_camera=rotation,
        t_rig_to_camera=-rotation @ center,
    )


def _local_matches(seed: int = 4):
    rng = np.random.default_rng(seed)
    front = rng.uniform((260.0, 180.0), (380.0, 300.0), size=(80, 2))
    affine = np.array(((1.02, 0.03, 36.0), (0.0, 1.0, 0.0)))
    side = front @ affine[:, :2].T + affine[:, 2]
    side += rng.normal(0.0, 0.35, size=side.shape)
    side[:12] = rng.uniform((40.0, 30.0), (600.0, 450.0), size=(12, 2))
    confidence = np.full(len(front), 0.9, dtype=np.float64)
    return front, side, confidence, affine


def test_local_affine_maps_exact_front_semantic_to_side_material_point():
    front, side, confidence, affine = _local_matches()
    anchor = np.array((320.0, 240.0))
    expected = affine[:, :2] @ anchor + affine[:, 2]

    result = refine_semantic_pair(
        anchor,
        expected + np.array((15.0, 4.0)),
        front,
        side,
        confidence,
        _camera("camera2", "front", 0.0),
        _camera("camera1", "left", -0.1),
        thresholds=SemanticRefinementThresholds(
            max_epipolar_correction_px=6.0,
            min_inlier_matches=6,
        ),
    )

    assert result["passed"] is True
    assert result["inlier_ratio"] > 0.7
    assert result["corrected_side_undistorted_px"] == pytest.approx(
        expected, abs=1.5
    )


def test_refinement_rejects_semantic_anchor_without_local_image_support():
    front, side, confidence, _affine = _local_matches()

    result = refine_semantic_pair(
        np.array((40.0, 40.0)),
        np.array((80.0, 40.0)),
        front,
        side,
        confidence,
        _camera("camera2", "front", 0.0),
        _camera("camera1", "left", -0.1),
    )

    assert result["passed"] is False
    assert "insufficient_local_image_matches" in result["issues"]
    assert "semantic_anchor_not_locally_supported" in result["issues"]


def test_refinement_rejects_prediction_far_from_semantic_region():
    front, side, confidence, _affine = _local_matches()

    result = refine_semantic_pair(
        np.array((320.0, 240.0)),
        np.array((80.0, 240.0)),
        front,
        side,
        confidence,
        _camera("camera2", "front", 0.0),
        _camera("camera1", "left", -0.1),
        thresholds=SemanticRefinementThresholds(
            side_prior_radius_px=400.0,
            max_detector_shift_px=30.0,
        ),
    )

    assert result["passed"] is False
    assert "refined_point_left_semantic_roi" in result["issues"]


def test_local_depth_surface_recovers_anchor_depth_with_outliers():
    rng = np.random.default_rng(17)
    anchor = np.array((320.0, 240.0))
    points = rng.uniform((275.0, 195.0), (365.0, 285.0), size=(90, 2))
    offsets = points - anchor
    depth = 0.15 + offsets[:, 0] * 0.00012 - offsets[:, 1] * 0.00008
    depth += rng.normal(0.0, 0.0004, len(depth))
    depth[:15] += rng.normal(0.03, 0.01, 15)
    confidence = np.full(len(depth), 0.9, dtype=np.float64)

    result = estimate_local_depth_surface(
        anchor,
        points,
        depth,
        confidence,
    )

    assert result["passed"] is True
    assert result["depth_m"] == pytest.approx(0.15, abs=0.0015)
    assert result["residual_p90_m"] < 0.002

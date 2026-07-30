from __future__ import annotations

import numpy as np
import pytest

from src.geometry.profile_shape_fit import (
    ProfileShapeFitConfig,
    _minimum_drift_update,
    profile_target_from_silhouette_report,
)


def test_minimum_drift_update_hits_metric_inside_trust_region():
    row = np.array([2.0, 0.0, 0.0])
    protected = np.array(
        [
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ]
    )

    result = _minimum_drift_update(
        row,
        0.008,
        protected,
        ridge=1e-6,
        max_l2=1.0,
        max_abs=1.0,
    )

    assert result["achieved_delta_m"] == pytest.approx(0.008)
    assert result["update"][1:] == pytest.approx([0.0, 0.0])
    assert result["trust_scale"] == pytest.approx(1.0)


def test_minimum_drift_update_reports_trust_region_clipping():
    result = _minimum_drift_update(
        np.array([1.0, 0.0]),
        4.0,
        np.eye(2),
        ridge=1e-6,
        max_l2=0.5,
        max_abs=0.5,
    )

    assert result["trust_scale"] < 1.0
    assert np.linalg.norm(result["update"]) == pytest.approx(0.5)
    assert result["achieved_delta_m"] == pytest.approx(0.5)


def test_profile_target_requires_two_side_nose_support():
    report = {
        "comparison_to_local_depth_surface": {
            "nose_tip_minus_upper_lip_mm": 18.0,
        },
        "points": {
            "nose_tip": {
                "passed": True,
                "valid_side_count": 2,
            }
        },
    }
    assert profile_target_from_silhouette_report(report) == pytest.approx(0.018)

    report["points"]["nose_tip"]["valid_side_count"] = 1
    with pytest.raises(ValueError, match="two-side"):
        profile_target_from_silhouette_report(report)


def test_profile_shape_defaults_keep_small_observation_changes_deadbanded():
    assert ProfileShapeFitConfig().target_deadband_m == pytest.approx(0.0015)

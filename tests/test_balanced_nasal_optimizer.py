from __future__ import annotations

from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import src.geometry.balanced_nasal_optimizer as balanced
from src.geometry.multiview_nasal_objective import (
    MultiviewNasalObjectiveConfig,
)
from src.geometry.nasal_semantic_basis import NASAL_SEMANTIC_MODE_NAMES


def _context() -> SimpleNamespace:
    terms = (
        SimpleNamespace(name="front_left", semantic_view="front"),
        SimpleNamespace(name="front_right", semantic_view="front"),
        SimpleNamespace(name="left_profile", semantic_view="subject-left"),
        SimpleNamespace(name="right_profile", semantic_view="subject-right"),
    )
    return SimpleNamespace(
        parameter_count=len(NASAL_SEMANTIC_MODE_NAMES),
        observable_rank=0,
        parameter_ordering=NASAL_SEMANTIC_MODE_NAMES,
        image_terms=terms,
    )


def test_baseline_calibration_equalizes_front_and_profile_robust_cost(
    monkeypatch,
):
    context = _context()
    term_residuals = MappingProxyType(
        {
            "front_left": np.asarray((2.0, 3.0)),
            "front_right": np.asarray((1.0, 2.0)),
            "left_profile": np.asarray((18.0, 25.0)),
            "right_profile": np.asarray((20.0, 30.0)),
        }
    )

    def evaluate(_coefficients, _context, _config):
        return SimpleNamespace(term_residuals=term_residuals)

    monkeypatch.setattr(
        balanced,
        "evaluate_multiview_nasal_objective_residuals",
        evaluate,
    )

    calibrated = balanced.calibrate_balanced_view_weights(
        context,
        MultiviewNasalObjectiveConfig(),
    )

    assert calibrated.front_weight == 1.0
    assert 0.0 < calibrated.side_weight < 1.0
    assert calibrated.initial_side_robust_cost > (
        calibrated.initial_front_robust_cost
    )
    assert calibrated.balanced_front_robust_cost == pytest.approx(
        calibrated.balanced_side_robust_cost,
        rel=1e-9,
        abs=1e-9,
    )


def test_staged_fit_uses_only_semantic_modes_and_declared_ownership(
    monkeypatch,
):
    context = _context()
    calls = []

    monkeypatch.setattr(
        balanced,
        "calibrate_balanced_view_weights",
        lambda *_args, **_kwargs: balanced.BalancedNasalViewWeights(
            front_weight=1.0,
            side_weight=0.2,
            initial_front_robust_cost=5.0,
            initial_side_robust_cost=25.0,
            balanced_front_robust_cost=5.0,
            balanced_side_robust_cost=5.0,
        ),
    )

    def fit(
        _context,
        objective_config,
        _optimization_config,
        initial_coefficients,
        *,
        active_parameter_indices,
    ):
        coefficients = np.asarray(initial_coefficients).copy()
        coefficients[np.asarray(active_parameter_indices)] += len(calls) + 1
        result = SimpleNamespace(
            success=True,
            coefficients=coefficients,
            final_objective=SimpleNamespace(candidate=SimpleNamespace()),
            failure_reason=None,
        )
        calls.append(
            {
                "front_weight": objective_config.front_image_weight,
                "side_weight": objective_config.side_image_weight,
                "active": tuple(active_parameter_indices),
                "initial": np.asarray(initial_coefficients).copy(),
                "result": result,
            }
        )
        return result

    monkeypatch.setattr(balanced, "fit_multiview_nasal_shape", fit)

    result = balanced.fit_balanced_multiview_nasal_shape(context)

    index = {name: offset for offset, name in enumerate(NASAL_SEMANTIC_MODE_NAMES)}
    assert len(calls) == 3
    assert calls[0]["front_weight"] == 1.0
    assert calls[0]["side_weight"] == 0.0
    assert calls[0]["active"] == (
        index["alar_width_shared"],
        index["alar_width_asymmetry"],
        index["tip_vertical"],
    )
    assert calls[1]["front_weight"] == 0.0
    assert calls[1]["side_weight"] == 1.0
    assert calls[1]["active"] == (
        index["alar_depth_shared"],
        index["alar_depth_asymmetry"],
        index["tip_depth"],
        index["tip_roundness"],
        index["tip_alar_fullness"],
    )
    assert calls[2]["front_weight"] == 1.0
    assert calls[2]["side_weight"] == pytest.approx(0.2)
    assert calls[2]["active"] == tuple(range(8))
    np.testing.assert_array_equal(calls[1]["initial"], calls[0]["result"].coefficients)
    np.testing.assert_array_equal(calls[2]["initial"], calls[1]["result"].coefficients)
    assert result.success
    assert result.final_result is calls[2]["result"]


def test_staged_fit_rejects_observable_flame_parameters():
    context = _context()
    context.observable_rank = 1
    context.parameter_ordering = ("observable_flame_0",) + (
        NASAL_SEMANTIC_MODE_NAMES
    )
    context.parameter_count = len(context.parameter_ordering)

    with pytest.raises(ValueError, match="exactly the eight semantic"):
        balanced.fit_balanced_multiview_nasal_shape(context)

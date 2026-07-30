"""Staged semantic-only nasal optimization with balanced view evidence."""

from __future__ import annotations

from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import Mapping, Optional

import numpy as np

from src.geometry.multiview_nasal_objective import (
    MultiviewNasalObjectiveConfig,
    evaluate_multiview_nasal_objective_residuals,
)
from src.geometry.multiview_nasal_optimizer import (
    NasalOptimizationConfig,
    NasalOptimizationResult,
    fit_multiview_nasal_shape,
)
from src.geometry.nasal_semantic_basis import NASAL_SEMANTIC_MODE_NAMES


_FRONT_STAGE_NAMES = (
    "alar_width_shared",
    "alar_width_asymmetry",
    "tip_vertical",
)
_PROFILE_STAGE_NAMES = (
    "alar_depth_shared",
    "alar_depth_asymmetry",
    "tip_depth",
    "tip_roundness",
    "tip_alar_fullness",
)


def _soft_l1_cost(residuals: np.ndarray, f_scale: float) -> float:
    values = np.asarray(residuals, dtype=np.float64) / float(f_scale)
    return float(
        float(f_scale) ** 2
        * np.sum(np.sqrt(1.0 + values * values) - 1.0)
    )


def _group_robust_cost(
    residuals: tuple[np.ndarray, ...],
    scale: float,
    f_scale: float,
) -> float:
    return float(
        sum(
            _soft_l1_cost(float(scale) * values, f_scale)
            for values in residuals
        )
    )


@dataclass(frozen=True)
class BalancedNasalViewWeights:
    """Baseline-calibrated weights and their diagnostic robust costs."""

    front_weight: float
    side_weight: float
    initial_front_robust_cost: float
    initial_side_robust_cost: float
    balanced_front_robust_cost: float
    balanced_side_robust_cost: float

    def __post_init__(self) -> None:
        values = np.asarray(
            (
                self.front_weight,
                self.side_weight,
                self.initial_front_robust_cost,
                self.initial_side_robust_cost,
                self.balanced_front_robust_cost,
                self.balanced_side_robust_cost,
            ),
            dtype=np.float64,
        )
        if (
            not np.isfinite(values).all()
            or self.front_weight <= 0.0
            or self.side_weight <= 0.0
            or np.any(values[2:] < 0.0)
        ):
            raise ValueError(
                "balanced nasal view weights and costs must be finite "
                "with positive weights and non-negative costs"
            )

    def to_report_data(self) -> dict[str, float]:
        return {
            "front_weight": float(self.front_weight),
            "side_weight": float(self.side_weight),
            "initial_front_robust_cost": float(
                self.initial_front_robust_cost
            ),
            "initial_side_robust_cost": float(
                self.initial_side_robust_cost
            ),
            "balanced_front_robust_cost": float(
                self.balanced_front_robust_cost
            ),
            "balanced_side_robust_cost": float(
                self.balanced_side_robust_cost
            ),
        }


@dataclass(frozen=True)
class BalancedNasalOptimizationResult:
    """Three ordered optimization stages and the final joint result."""

    success: bool
    view_weights: BalancedNasalViewWeights
    stage_results: Mapping[str, NasalOptimizationResult]
    final_result: NasalOptimizationResult
    failure_stage: Optional[str] = None

    def __post_init__(self) -> None:
        stages = dict(self.stage_results)
        expected = ("front", "profile", "joint")
        if tuple(stages) not in (expected, expected[:1], expected[:2]):
            raise ValueError("balanced nasal stages are not in canonical order")
        if self.success:
            if tuple(stages) != expected or self.failure_stage is not None:
                raise ValueError(
                    "successful balanced fit requires all three stages"
                )
        elif self.failure_stage not in expected:
            raise ValueError("failed balanced fit needs a canonical stage")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(
            self,
            "stage_results",
            MappingProxyType(stages),
        )

    @property
    def coefficients(self) -> np.ndarray:
        return self.final_result.coefficients

    @property
    def final_objective(self):
        return self.final_result.final_objective

    def to_report_data(self) -> dict[str, object]:
        return {
            "success": bool(self.success),
            "failure_stage": self.failure_stage,
            "view_balance": self.view_weights.to_report_data(),
            "stages": {
                name: {
                    "success": bool(result.success),
                    "failure_reason": result.failure_reason,
                    "coefficients": [
                        float(value) for value in result.coefficients
                    ],
                    "active_parameter_indices": list(
                        result.report.get("active_parameter_indices", ())
                    ),
                    "active_parameter_names": list(
                        result.report.get("active_parameter_names", ())
                    ),
                }
                for name, result in self.stage_results.items()
            },
        }


def calibrate_balanced_view_weights(
    context,
    objective_config: Optional[MultiviewNasalObjectiveConfig] = None,
    coefficients: Optional[np.ndarray] = None,
) -> BalancedNasalViewWeights:
    """Equalize front-group and two-profile-group robust cost at baseline."""
    base = (
        MultiviewNasalObjectiveConfig()
        if objective_config is None
        else objective_config
    )
    if not isinstance(base, MultiviewNasalObjectiveConfig):
        raise ValueError(
            "objective_config must be a MultiviewNasalObjectiveConfig"
        )
    theta = (
        np.zeros(int(context.parameter_count), dtype=np.float64)
        if coefficients is None
        else np.asarray(coefficients, dtype=np.float64)
    )
    if (
        theta.shape != (int(context.parameter_count),)
        or not np.isfinite(theta).all()
    ):
        raise ValueError("coefficients must be a finite parameter vector")
    unit_config = replace(
        base,
        front_image_weight=1.0,
        side_image_weight=1.0,
    )
    evaluated = evaluate_multiview_nasal_objective_residuals(
        theta,
        context,
        unit_config,
    )
    terms_by_name = {
        str(term.name): str(term.semantic_view)
        for term in context.image_terms
    }
    front = tuple(
        np.asarray(values, dtype=np.float64)
        for name, values in evaluated.term_residuals.items()
        if terms_by_name.get(str(name)) == "front"
    )
    side = tuple(
        np.asarray(values, dtype=np.float64)
        for name, values in evaluated.term_residuals.items()
        if terms_by_name.get(str(name)) in ("subject-left", "subject-right")
    )
    if not front or not side:
        raise ValueError(
            "balanced calibration requires front and bilateral profile terms"
        )
    f_scale = float(unit_config.robust_f_scale)
    front_cost = _group_robust_cost(front, 1.0, f_scale)
    side_cost = _group_robust_cost(side, 1.0, f_scale)
    if front_cost <= 0.0 or side_cost <= 0.0:
        raise ValueError(
            "balanced calibration requires positive baseline image costs"
        )

    low = 0.0
    high = 1.0
    while _group_robust_cost(side, high, f_scale) < front_cost:
        high *= 2.0
        if high > 1e6:
            raise RuntimeError("could not bracket balanced side-view weight")
    for _ in range(80):
        middle = 0.5 * (low + high)
        if _group_robust_cost(side, middle, f_scale) < front_cost:
            low = middle
        else:
            high = middle
    side_weight = 0.5 * (low + high)
    balanced_side = _group_robust_cost(side, side_weight, f_scale)
    return BalancedNasalViewWeights(
        front_weight=1.0,
        side_weight=side_weight,
        initial_front_robust_cost=front_cost,
        initial_side_robust_cost=side_cost,
        balanced_front_robust_cost=front_cost,
        balanced_side_robust_cost=balanced_side,
    )


def fit_balanced_multiview_nasal_shape(
    context,
    objective_config: Optional[MultiviewNasalObjectiveConfig] = None,
    optimization_config: Optional[NasalOptimizationConfig] = None,
    initial_coefficients: Optional[np.ndarray] = None,
) -> BalancedNasalOptimizationResult:
    """Fit front, profile, then balanced joint semantic nasal parameters."""
    if (
        int(getattr(context, "observable_rank", -1)) != 0
        or tuple(getattr(context, "parameter_ordering", ()))
        != tuple(NASAL_SEMANTIC_MODE_NAMES)
        or int(getattr(context, "parameter_count", -1))
        != len(NASAL_SEMANTIC_MODE_NAMES)
    ):
        raise ValueError(
            "balanced nasal fitting requires exactly the eight semantic "
            "parameters and no observable FLAME directions"
        )
    base = (
        MultiviewNasalObjectiveConfig()
        if objective_config is None
        else objective_config
    )
    if not isinstance(base, MultiviewNasalObjectiveConfig):
        raise ValueError(
            "objective_config must be a MultiviewNasalObjectiveConfig"
        )
    initial = (
        np.zeros(len(NASAL_SEMANTIC_MODE_NAMES), dtype=np.float64)
        if initial_coefficients is None
        else np.asarray(initial_coefficients, dtype=np.float64)
    )
    if (
        initial.shape != (len(NASAL_SEMANTIC_MODE_NAMES),)
        or not np.isfinite(initial).all()
    ):
        raise ValueError("initial_coefficients must be a finite 8-vector")
    mode_index = {
        name: index for index, name in enumerate(NASAL_SEMANTIC_MODE_NAMES)
    }
    balance = calibrate_balanced_view_weights(context, base, initial)
    stage_specs = (
        (
            "front",
            replace(base, front_image_weight=1.0, side_image_weight=0.0),
            tuple(mode_index[name] for name in _FRONT_STAGE_NAMES),
        ),
        (
            "profile",
            replace(base, front_image_weight=0.0, side_image_weight=1.0),
            tuple(mode_index[name] for name in _PROFILE_STAGE_NAMES),
        ),
        (
            "joint",
            replace(
                base,
                front_image_weight=balance.front_weight,
                side_image_weight=balance.side_weight,
            ),
            tuple(range(len(NASAL_SEMANTIC_MODE_NAMES))),
        ),
    )
    stage_results = {}
    coefficients = np.array(initial, dtype=np.float64, copy=True)
    for name, stage_config, active in stage_specs:
        result = fit_multiview_nasal_shape(
            context,
            stage_config,
            optimization_config,
            coefficients,
            active_parameter_indices=np.asarray(active, dtype=np.int64),
        )
        stage_results[name] = result
        coefficients = np.asarray(result.coefficients, dtype=np.float64)
        if not result.success:
            return BalancedNasalOptimizationResult(
                success=False,
                view_weights=balance,
                stage_results=stage_results,
                final_result=result,
                failure_stage=name,
            )
    return BalancedNasalOptimizationResult(
        success=True,
        view_weights=balance,
        stage_results=stage_results,
        final_result=stage_results["joint"],
        failure_stage=None,
    )


__all__ = [
    "BalancedNasalOptimizationResult",
    "BalancedNasalViewWeights",
    "calibrate_balanced_view_weights",
    "fit_balanced_multiview_nasal_shape",
]

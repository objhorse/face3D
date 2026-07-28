"""Bounded robust optimization for the unified multiview nasal objective."""

from __future__ import annotations

from dataclasses import dataclass
from numbers import Integral, Real
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

import numpy as np
from scipy.optimize import least_squares

from src.geometry.multiview_nasal_objective import (
    MultiviewNasalObjectiveConfig,
    MultiviewNasalObjectiveContext,
    MultiviewNasalObjectiveResult,
    evaluate_multiview_nasal_objective,
    evaluate_multiview_nasal_objective_residuals,
)


__all__ = [
    "NasalOptimizationConfig",
    "NasalOptimizationIteration",
    "NasalOptimizationResult",
    "fit_multiview_nasal_shape",
]

_FAILURE_REASONS = frozenset(
    {
        "invalid_initial_coefficients",
        "no_effective_observations",
        "objective_evaluation_failed",
        "solver_failed",
        "singular_jacobian",
    }
)


def _readonly_array(value, dtype=None) -> np.ndarray:
    array = np.ascontiguousarray(value, dtype=dtype)
    return np.frombuffer(
        array.tobytes(order="C"),
        dtype=array.dtype,
        count=array.size,
    ).reshape(array.shape)


def _finite_real(
    name: str,
    value,
    *,
    strictly_positive: bool = False,
) -> float:
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, Real)
        or not np.isfinite(value)
    ):
        raise ValueError(f"{name} must be finite and numeric")
    result = float(value)
    if result < 0.0 or (strictly_positive and result <= 0.0):
        qualifier = "positive" if strictly_positive else "non-negative"
        raise ValueError(f"{name} must be {qualifier}")
    return result


def _freeze(value):
    if isinstance(value, np.ndarray):
        return _readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType(
            {str(key): _freeze(item) for key, item in value.items()}
        )
    if isinstance(value, (list, tuple)):
        return tuple(_freeze(item) for item in value)
    return value


@dataclass(frozen=True)
class NasalOptimizationConfig:
    """Subject-independent numerical convergence settings for SciPy TRF."""

    max_nfev: Optional[int] = 200
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    diff_step: Optional[float] = None
    trace_cost_tolerance: float = 1e-12
    method: str = "trf"
    jac: str = "2-point"

    def __post_init__(self) -> None:
        if self.max_nfev is not None and (
            isinstance(self.max_nfev, (bool, np.bool_))
            or not isinstance(self.max_nfev, Integral)
            or int(self.max_nfev) < 1
        ):
            raise ValueError("max_nfev must be None or a positive integer")
        for name in ("ftol", "xtol", "gtol"):
            _finite_real(name, getattr(self, name), strictly_positive=True)
        if self.diff_step is not None:
            _finite_real("diff_step", self.diff_step, strictly_positive=True)
        _finite_real("trace_cost_tolerance", self.trace_cost_tolerance)
        if str(self.method) != "trf":
            raise ValueError("method must be the fixed value 'trf'")
        if str(self.jac) != "2-point":
            raise ValueError("jac must be the fixed finite difference '2-point'")
        object.__setattr__(
            self,
            "max_nfev",
            None if self.max_nfev is None else int(self.max_nfev),
        )
        object.__setattr__(self, "method", "trf")
        object.__setattr__(self, "jac", "2-point")


@dataclass(frozen=True)
class NasalOptimizationIteration:
    """One immutable best-so-far objective observation."""

    function_evaluation: int
    coefficients: np.ndarray
    total_robust_cost: float
    robust_term_costs: Mapping[str, float]

    def __post_init__(self) -> None:
        if (
            isinstance(self.function_evaluation, (bool, np.bool_))
            or not isinstance(self.function_evaluation, Integral)
            or int(self.function_evaluation) < 1
        ):
            raise ValueError("function_evaluation must be a positive integer")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError("trace coefficients must be a finite vector")
        costs = {
            str(name): _finite_real(
                f"robust term cost {name}",
                value,
            )
            for name, value in self.robust_term_costs.items()
        }
        if not costs:
            raise ValueError("robust_term_costs must not be empty")
        total = _finite_real("total_robust_cost", self.total_robust_cost)
        if not np.isclose(
            total,
            sum(costs.values()),
            atol=1e-12,
            rtol=1e-12,
        ):
            raise ValueError("trace term costs must sum to total_robust_cost")
        object.__setattr__(
            self,
            "function_evaluation",
            int(self.function_evaluation),
        )
        object.__setattr__(
            self,
            "coefficients",
            _readonly_array(coefficients, np.float64),
        )
        object.__setattr__(
            self,
            "robust_term_costs",
            MappingProxyType(costs),
        )
        object.__setattr__(self, "total_robust_cost", total)


@dataclass(frozen=True)
class NasalOptimizationResult:
    """Complete immutable solver outcome; the source baseline is never mutated."""

    success: bool
    coefficients: np.ndarray
    final_objective: Optional[MultiviewNasalObjectiveResult]
    solver_status: int
    solver_message: str
    nfev: int
    njev: Optional[int]
    objective_evaluation_count: int
    optimality: Optional[float]
    active_mask: np.ndarray
    jacobian_rank: Optional[int]
    iteration_trace: Tuple[NasalOptimizationIteration, ...]
    objective_term_trace: Mapping[str, Tuple[float, ...]]
    report: Mapping[str, object]
    baseline_unchanged: bool = True
    failure_reason: Optional[str] = None

    def __post_init__(self) -> None:
        if not isinstance(self.success, (bool, np.bool_)):
            raise ValueError("success must be boolean")
        coefficients = np.asarray(self.coefficients, dtype=np.float64)
        if coefficients.ndim != 1 or not np.isfinite(coefficients).all():
            raise ValueError("result coefficients must be a finite vector")
        if self.final_objective is not None and not isinstance(
            self.final_objective,
            MultiviewNasalObjectiveResult,
        ):
            raise ValueError(
                "final_objective must be a MultiviewNasalObjectiveResult or None"
            )
        for name in ("solver_status", "nfev", "objective_evaluation_count"):
            value = getattr(self, name)
            if (
                isinstance(value, (bool, np.bool_))
                or not isinstance(value, Integral)
            ):
                raise ValueError(f"{name} must be an integer")
            if name != "solver_status" and int(value) < 0:
                raise ValueError(f"{name} must be non-negative")
        if self.njev is not None and (
            isinstance(self.njev, (bool, np.bool_))
            or not isinstance(self.njev, Integral)
            or int(self.njev) < 0
        ):
            raise ValueError("njev must be None or a non-negative integer")
        optimality = None
        if self.optimality is not None:
            optimality = _finite_real("optimality", self.optimality)
        active_mask = np.asarray(self.active_mask)
        if (
            active_mask.shape != coefficients.shape
            or not np.issubdtype(active_mask.dtype, np.integer)
            or np.any(~np.isin(active_mask, (-1, 0, 1)))
        ):
            raise ValueError("active_mask must contain -1, 0, or 1 per parameter")
        rank = self.jacobian_rank
        if rank is not None and (
            isinstance(rank, (bool, np.bool_))
            or not isinstance(rank, Integral)
            or int(rank) < 0
            or int(rank) > len(coefficients)
        ):
            raise ValueError("jacobian_rank is invalid")
        trace = tuple(self.iteration_trace)
        if not all(
            isinstance(item, NasalOptimizationIteration) for item in trace
        ):
            raise ValueError("iteration_trace contains an invalid entry")
        if any(
            len(item.coefficients) != len(coefficients) for item in trace
        ):
            raise ValueError("trace coefficient dimensions do not match result")
        if any(
            later.total_robust_cost > earlier.total_robust_cost + 1e-12
            for earlier, later in zip(trace, trace[1:])
        ):
            raise ValueError("iteration_trace must be monotonic non-increasing")
        term_trace = {
            str(name): tuple(
                _finite_real(f"objective term trace {name}", value)
                for value in values
            )
            for name, values in self.objective_term_trace.items()
        }
        expected_names = (
            tuple(trace[0].robust_term_costs) if trace else tuple()
        )
        if tuple(term_trace) != expected_names or any(
            len(values) != len(trace) for values in term_trace.values()
        ):
            raise ValueError("objective_term_trace does not match iteration_trace")
        for index, item in enumerate(trace):
            if any(
                term_trace[name][index] != item.robust_term_costs[name]
                for name in expected_names
            ):
                raise ValueError(
                    "objective_term_trace values do not match iteration_trace"
                )
        reason = self.failure_reason
        if bool(self.success):
            if reason is not None or self.final_objective is None:
                raise ValueError(
                    "successful result requires a final objective and no failure"
                )
        elif reason not in _FAILURE_REASONS:
            raise ValueError("failed result requires a canonical failure_reason")
        if self.baseline_unchanged is not True:
            raise ValueError("baseline_unchanged must be true")
        object.__setattr__(self, "success", bool(self.success))
        object.__setattr__(
            self,
            "coefficients",
            _readonly_array(coefficients, np.float64),
        )
        object.__setattr__(self, "solver_status", int(self.solver_status))
        object.__setattr__(self, "solver_message", str(self.solver_message))
        object.__setattr__(self, "nfev", int(self.nfev))
        object.__setattr__(
            self,
            "njev",
            None if self.njev is None else int(self.njev),
        )
        object.__setattr__(
            self,
            "objective_evaluation_count",
            int(self.objective_evaluation_count),
        )
        object.__setattr__(self, "optimality", optimality)
        object.__setattr__(
            self,
            "active_mask",
            _readonly_array(active_mask, np.int64),
        )
        object.__setattr__(
            self,
            "jacobian_rank",
            None if rank is None else int(rank),
        )
        object.__setattr__(self, "iteration_trace", trace)
        object.__setattr__(
            self,
            "objective_term_trace",
            MappingProxyType(term_trace),
        )
        object.__setattr__(self, "report", _freeze(self.report))

    @property
    def trace(self) -> Tuple[NasalOptimizationIteration, ...]:
        return self.iteration_trace


class _ObjectiveEvaluationError(RuntimeError):
    pass


def _soft_l1_cost(residuals: np.ndarray, f_scale: float) -> float:
    scaled = np.asarray(residuals, dtype=np.float64) / float(f_scale)
    return float(
        float(f_scale) ** 2
        * np.sum(np.sqrt(1.0 + scaled * scaled) - 1.0)
    )


def _objective_term_trace(
    trace: Tuple[NasalOptimizationIteration, ...],
) -> Mapping[str, Tuple[float, ...]]:
    if not trace:
        return {}
    return {
        name: tuple(item.robust_term_costs[name] for item in trace)
        for name in trace[0].robust_term_costs
    }


def _jacobian_rank(jacobian) -> Optional[int]:
    if jacobian is None:
        return None
    if hasattr(jacobian, "toarray"):
        jacobian = jacobian.toarray()
    array = np.asarray(jacobian, dtype=np.float64)
    if array.ndim != 2 or not np.isfinite(array).all():
        return None
    return int(np.linalg.matrix_rank(array))


def _safe_full_evaluation(
    coefficients: np.ndarray,
    context: MultiviewNasalObjectiveContext,
    config: MultiviewNasalObjectiveConfig,
) -> Tuple[Optional[MultiviewNasalObjectiveResult], Optional[str]]:
    try:
        result = evaluate_multiview_nasal_objective(
            coefficients,
            context,
            config,
        )
    except Exception as exc:  # Diagnostics must never obscure solver failure.
        return None, f"{type(exc).__name__}: {exc}"
    return result, None


def _result(
    *,
    success: bool,
    coefficients: np.ndarray,
    final_objective: Optional[MultiviewNasalObjectiveResult],
    failure_reason: Optional[str],
    solver_status: int,
    solver_message: str,
    nfev: int,
    njev: Optional[int],
    evaluation_count: int,
    optimality: Optional[float],
    active_mask: np.ndarray,
    jacobian_rank: Optional[int],
    trace,
    report,
) -> NasalOptimizationResult:
    frozen_trace = tuple(trace)
    return NasalOptimizationResult(
        success=success,
        coefficients=coefficients,
        final_objective=final_objective,
        solver_status=solver_status,
        solver_message=solver_message,
        nfev=nfev,
        njev=njev,
        objective_evaluation_count=evaluation_count,
        optimality=optimality,
        active_mask=active_mask,
        jacobian_rank=jacobian_rank,
        iteration_trace=frozen_trace,
        objective_term_trace=_objective_term_trace(frozen_trace),
        report=report,
        baseline_unchanged=True,
        failure_reason=failure_reason,
    )


def fit_multiview_nasal_shape(
    context: MultiviewNasalObjectiveContext,
    objective_config: Optional[MultiviewNasalObjectiveConfig] = None,
    optimization_config: Optional[NasalOptimizationConfig] = None,
    initial_coefficients: Optional[np.ndarray] = None,
) -> NasalOptimizationResult:
    """Fit one context without mutating or writing its baseline artifacts."""
    if not isinstance(context, MultiviewNasalObjectiveContext):
        raise ValueError("context must be a MultiviewNasalObjectiveContext")
    objective_limits = (
        MultiviewNasalObjectiveConfig()
        if objective_config is None
        else objective_config
    )
    if not isinstance(objective_limits, MultiviewNasalObjectiveConfig):
        raise ValueError(
            "objective_config must be a MultiviewNasalObjectiveConfig"
        )
    solver_limits = (
        NasalOptimizationConfig()
        if optimization_config is None
        else optimization_config
    )
    if not isinstance(solver_limits, NasalOptimizationConfig):
        raise ValueError(
            "optimization_config must be a NasalOptimizationConfig"
        )

    zero = np.zeros(context.parameter_count, dtype=np.float64)
    try:
        initial = (
            zero
            if initial_coefficients is None
            else np.asarray(initial_coefficients, dtype=np.float64)
        )
    except (TypeError, ValueError) as exc:
        return _result(
            success=False,
            coefficients=zero,
            final_objective=None,
            failure_reason="invalid_initial_coefficients",
            solver_status=0,
            solver_message=f"invalid initial coefficients: {exc}",
            nfev=0,
            njev=None,
            evaluation_count=0,
            optimality=None,
            active_mask=np.zeros_like(zero, dtype=np.int64),
            jacobian_rank=None,
            trace=(),
            report={"selected_coefficients_source": "baseline_zero"},
        )
    if (
        initial.shape != (context.parameter_count,)
        or not np.isfinite(initial).all()
        or np.any(initial < context.parameter_lower_bounds)
        or np.any(initial > context.parameter_upper_bounds)
    ):
        return _result(
            success=False,
            coefficients=zero,
            final_objective=None,
            failure_reason="invalid_initial_coefficients",
            solver_status=0,
            solver_message=(
                "initial coefficients must be a finite vector inside "
                "the context bounds"
            ),
            nfev=0,
            njev=None,
            evaluation_count=0,
            optimality=None,
            active_mask=np.zeros_like(zero, dtype=np.int64),
            jacobian_rank=None,
            trace=(),
            report={"selected_coefficients_source": "baseline_zero"},
        )
    initial = np.array(initial, dtype=np.float64, copy=True)

    evidence_counts = {
        term.name: int(np.count_nonzero(term.confidence > 0.0))
        for term in context.image_terms
    }
    evidence_sums = {
        term.name: float(np.sum(term.confidence))
        for term in context.image_terms
    }
    if sum(evidence_counts.values()) == 0:
        return _result(
            success=False,
            coefficients=initial,
            final_objective=None,
            failure_reason="no_effective_observations",
            solver_status=0,
            solver_message="no image term has positive-confidence evidence",
            nfev=0,
            njev=None,
            evaluation_count=0,
            optimality=None,
            active_mask=np.zeros_like(initial, dtype=np.int64),
            jacobian_rank=None,
            trace=(),
            report={
                "effective_observation_counts": evidence_counts,
                "effective_confidence_sums": evidence_sums,
                "selected_coefficients_source": "initial",
            },
        )

    x_scale = np.r_[
        context.flame_mode_standard_deviations,
        np.asarray(
            objective_limits.semantic_prior_standard_deviations,
            dtype=np.float64,
        ),
    ]
    trace = []
    evaluation_count = 0
    best_coefficients = initial.copy()
    best_total_cost = np.inf
    best_residuals = None

    def residual(coefficients):
        nonlocal evaluation_count
        nonlocal best_coefficients
        nonlocal best_total_cost
        nonlocal best_residuals
        evaluation_count += 1
        try:
            evaluated = evaluate_multiview_nasal_objective_residuals(
                coefficients,
                context,
                objective_limits,
            )
            residuals = np.asarray(evaluated.residuals, dtype=np.float64)
            if residuals.ndim != 1 or not np.isfinite(residuals).all():
                raise ValueError("objective returned non-finite residuals")
            term_costs = {
                str(name): _soft_l1_cost(
                    term_residuals,
                    objective_limits.robust_f_scale,
                )
                for name, term_residuals in evaluated.term_residuals.items()
            }
            if not term_costs:
                raise ValueError("objective returned no canonical terms")
            total_cost = float(sum(term_costs.values()))
            if not np.isfinite(total_cost):
                raise ValueError("objective returned non-finite robust cost")
        except Exception as exc:
            raise _ObjectiveEvaluationError(
                f"{type(exc).__name__}: {exc}"
            ) from exc
        improved = total_cost < best_total_cost
        if improved:
            best_coefficients = np.array(
                coefficients,
                dtype=np.float64,
                copy=True,
            )
            best_total_cost = total_cost
            best_residuals = np.array(residuals, copy=True)
        trace_improved = (
            not trace
            or total_cost
            < trace[-1].total_robust_cost
            - float(solver_limits.trace_cost_tolerance)
        )
        if trace_improved:
            trace.append(
                NasalOptimizationIteration(
                    function_evaluation=evaluation_count,
                    coefficients=best_coefficients,
                    total_robust_cost=total_cost,
                    robust_term_costs=term_costs,
                )
            )
        return np.array(residuals, copy=True)

    common_report = {
        "method": solver_limits.method,
        "jac": solver_limits.jac,
        "bounds": {
            "lower": tuple(context.parameter_lower_bounds),
            "upper": tuple(context.parameter_upper_bounds),
        },
        "loss": objective_limits.robust_loss,
        "f_scale": float(objective_limits.robust_f_scale),
        "x_scale": tuple(float(value) for value in x_scale),
        "effective_observation_counts": evidence_counts,
        "effective_confidence_sums": evidence_sums,
    }
    try:
        solved = least_squares(
            residual,
            initial,
            jac=solver_limits.jac,
            bounds=(
                context.parameter_lower_bounds,
                context.parameter_upper_bounds,
            ),
            method=solver_limits.method,
            ftol=float(solver_limits.ftol),
            xtol=float(solver_limits.xtol),
            gtol=float(solver_limits.gtol),
            x_scale=x_scale,
            loss=objective_limits.robust_loss,
            f_scale=float(objective_limits.robust_f_scale),
            diff_step=solver_limits.diff_step,
            max_nfev=solver_limits.max_nfev,
        )
        residual(np.asarray(solved.x, dtype=np.float64))
    except _ObjectiveEvaluationError as exc:
        diagnostic, diagnostic_error = _safe_full_evaluation(
            best_coefficients,
            context,
            objective_limits,
        )
        report = dict(common_report)
        report.update(
            {
                "objective_error": str(exc),
                "diagnostic_error": diagnostic_error,
                "selected_coefficients_source": (
                    "best_observed" if trace else "initial"
                ),
            }
        )
        return _result(
            success=False,
            coefficients=best_coefficients,
            final_objective=diagnostic,
            failure_reason="objective_evaluation_failed",
            solver_status=0,
            solver_message=f"objective evaluation failed: {exc}",
            nfev=0,
            njev=None,
            evaluation_count=evaluation_count,
            optimality=None,
            active_mask=np.zeros_like(initial, dtype=np.int64),
            jacobian_rank=None,
            trace=trace,
            report=report,
        )
    except Exception as exc:
        diagnostic, diagnostic_error = _safe_full_evaluation(
            best_coefficients,
            context,
            objective_limits,
        )
        report = dict(common_report)
        report.update(
            {
                "solver_error": f"{type(exc).__name__}: {exc}",
                "diagnostic_error": diagnostic_error,
                "selected_coefficients_source": (
                    "best_observed" if trace else "initial"
                ),
            }
        )
        return _result(
            success=False,
            coefficients=best_coefficients,
            final_objective=diagnostic,
            failure_reason="solver_failed",
            solver_status=0,
            solver_message=f"solver failed: {type(exc).__name__}: {exc}",
            nfev=0,
            njev=None,
            evaluation_count=evaluation_count,
            optimality=None,
            active_mask=np.zeros_like(initial, dtype=np.int64),
            jacobian_rank=None,
            trace=trace,
            report=report,
        )

    selected = best_coefficients
    solver_x = np.asarray(solved.x, dtype=np.float64)
    selected_source = (
        "solver_final"
        if np.array_equal(selected, solver_x)
        else "best_observed"
    )
    final_objective, diagnostic_error = _safe_full_evaluation(
        selected,
        context,
        objective_limits,
    )
    status = int(getattr(solved, "status", 0))
    message = str(getattr(solved, "message", ""))
    nfev = int(getattr(solved, "nfev", 0))
    raw_njev = getattr(solved, "njev", None)
    njev = None if raw_njev is None else int(raw_njev)
    raw_optimality = getattr(solved, "optimality", None)
    optimality = (
        None if raw_optimality is None else float(raw_optimality)
    )
    active_mask = np.asarray(
        getattr(
            solved,
            "active_mask",
            np.zeros(context.parameter_count, dtype=np.int64),
        ),
        dtype=np.int64,
    )
    rank = _jacobian_rank(getattr(solved, "jac", None))
    report = dict(common_report)
    report.update(
        {
            "selected_coefficients_source": selected_source,
            "solver_success": bool(getattr(solved, "success", False)),
            "solver_final_coefficients": tuple(float(value) for value in solver_x),
            "best_observed_robust_cost": float(best_total_cost),
            "diagnostic_error": diagnostic_error,
            "jacobian_rank": rank,
            "parameter_count": context.parameter_count,
        }
    )
    if final_objective is None:
        return _result(
            success=False,
            coefficients=selected,
            final_objective=None,
            failure_reason="objective_evaluation_failed",
            solver_status=status,
            solver_message=(
                "final objective evaluation failed: "
                f"{diagnostic_error}"
            ),
            nfev=nfev,
            njev=njev,
            evaluation_count=evaluation_count,
            optimality=optimality,
            active_mask=active_mask,
            jacobian_rank=rank,
            trace=trace,
            report=report,
        )
    if not bool(getattr(solved, "success", False)):
        return _result(
            success=False,
            coefficients=selected,
            final_objective=final_objective,
            failure_reason="solver_failed",
            solver_status=status,
            solver_message=message,
            nfev=nfev,
            njev=njev,
            evaluation_count=evaluation_count,
            optimality=optimality,
            active_mask=active_mask,
            jacobian_rank=rank,
            trace=trace,
            report=report,
        )
    exact_zero_residual = (
        best_residuals is not None
        and np.count_nonzero(best_residuals) == 0
    )
    if rank is None or (
        rank < context.parameter_count and not exact_zero_residual
    ):
        return _result(
            success=False,
            coefficients=selected,
            final_objective=final_objective,
            failure_reason="singular_jacobian",
            solver_status=status,
            solver_message=(
                "solver converged but the finite-difference Jacobian "
                f"rank is {rank!r} for {context.parameter_count} parameters"
            ),
            nfev=nfev,
            njev=njev,
            evaluation_count=evaluation_count,
            optimality=optimality,
            active_mask=active_mask,
            jacobian_rank=rank,
            trace=trace,
            report=report,
        )
    return _result(
        success=True,
        coefficients=selected,
        final_objective=final_objective,
        failure_reason=None,
        solver_status=status,
        solver_message=message,
        nfev=nfev,
        njev=njev,
        evaluation_count=evaluation_count,
        optimality=optimality,
        active_mask=active_mask,
        jacobian_rank=rank,
        trace=trace,
        report=report,
    )

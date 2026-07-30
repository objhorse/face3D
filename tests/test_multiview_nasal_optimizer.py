from __future__ import annotations

from dataclasses import FrozenInstanceError
from types import MappingProxyType, SimpleNamespace

import numpy as np
import pytest

import src.geometry.multiview_nasal_optimizer as nasal_optimizer
from src.geometry.multiview_nasal_objective import (
    MultiviewNasalObjectiveConfig,
    evaluate_multiview_nasal_objective,
    evaluate_multiview_nasal_objective_residuals,
)
from src.geometry.multiview_nasal_optimizer import (
    NasalOptimizationConfig,
    NasalOptimizationIteration,
    fit_multiview_nasal_shape,
)
from tests.test_multiview_nasal_objective import (
    _objective_problem,
    _single_face_flip_semantic,
)


def _synthetic_evaluator(context, residual_function):
    template = evaluate_multiview_nasal_objective_residuals(
        np.zeros(context.parameter_count),
        context,
    )
    names = tuple(template.term_residuals)

    def evaluate(coefficients, _context, _config):
        residuals = np.asarray(
            residual_function(np.asarray(coefficients, dtype=np.float64)),
            dtype=np.float64,
        )
        terms = {names[0]: residuals}
        terms.update(
            {
                name: np.empty(0, dtype=np.float64)
                for name in names[1:]
            }
        )
        return SimpleNamespace(
            residuals=residuals,
            term_residuals=MappingProxyType(terms),
        )

    return evaluate


def _solver_result(context, **overrides):
    count = context.parameter_count
    values = {
        "x": np.zeros(count, dtype=np.float64),
        "success": True,
        "status": 1,
        "message": "synthetic solver result",
        "nfev": 1,
        "njev": 1,
        "optimality": 0.0,
        "active_mask": np.zeros(count, dtype=np.int64),
        "jac": np.eye(count, dtype=np.float64),
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def test_default_zero_initialization_keeps_exact_zero_solution(monkeypatch):
    context, _ = _objective_problem()
    baseline = np.array(context.baseline_vertices, copy=True)
    residual_calls = 0
    full_calls = 0
    real_full = evaluate_multiview_nasal_objective
    synthetic = _synthetic_evaluator(
        context,
        lambda coefficients: np.zeros_like(coefficients),
    )

    def counted_residual(*args):
        nonlocal residual_calls
        residual_calls += 1
        return synthetic(*args)

    def counted_full(*args):
        nonlocal full_calls
        full_calls += 1
        return real_full(*args)

    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        counted_residual,
    )
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective",
        counted_full,
    )

    result = fit_multiview_nasal_shape(context)

    assert result.success
    np.testing.assert_array_equal(
        result.coefficients,
        np.zeros(context.parameter_count),
    )
    np.testing.assert_array_equal(context.baseline_vertices, baseline)
    assert result.jacobian_rank == 0
    assert result.baseline_unchanged is True
    assert residual_calls == result.objective_evaluation_count
    assert residual_calls > full_calls
    assert full_calls == 1


def test_two_small_initializations_converge_to_same_synthetic_solution(
    monkeypatch,
):
    context, _ = _objective_problem()
    target = np.linspace(-0.35, 0.35, context.parameter_count)
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        _synthetic_evaluator(
            context,
            lambda coefficients: coefficients - target,
        ),
    )

    first = fit_multiview_nasal_shape(
        context,
        initial_coefficients=np.full(context.parameter_count, 0.05),
    )
    second = fit_multiview_nasal_shape(
        context,
        initial_coefficients=np.full(context.parameter_count, -0.08),
    )

    assert first.success
    assert second.success
    np.testing.assert_allclose(first.coefficients, target, atol=1e-8)
    np.testing.assert_allclose(second.coefficients, target, atol=1e-8)
    np.testing.assert_allclose(
        first.coefficients,
        second.coefficients,
        atol=1e-9,
    )


def test_active_parameter_subset_keeps_all_other_coefficients_fixed(
    monkeypatch,
):
    context, _ = _objective_problem()
    target = np.linspace(-0.35, 0.35, context.parameter_count)
    initial = np.linspace(0.12, -0.12, context.parameter_count)
    active = np.asarray((1, 4, 7), dtype=np.int64)
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        _synthetic_evaluator(
            context,
            lambda coefficients: coefficients - target,
        ),
    )

    result = fit_multiview_nasal_shape(
        context,
        initial_coefficients=initial,
        active_parameter_indices=active,
    )

    assert result.success
    expected = initial.copy()
    expected[active] = target[active]
    np.testing.assert_allclose(result.coefficients, expected, atol=1e-8)
    inactive = np.setdiff1d(np.arange(context.parameter_count), active)
    np.testing.assert_array_equal(
        result.coefficients[inactive],
        initial[inactive],
    )
    assert result.jacobian_rank == len(active)
    assert result.report["active_parameter_indices"] == tuple(active)
    assert result.report["active_parameter_names"] == tuple(
        context.parameter_ordering[index] for index in active
    )


def test_real_image_target_converges_without_flipping_an_active_face():
    target = np.zeros(8, dtype=np.float64)
    target[0] = 1.0
    context, _ = _objective_problem(
        target_semantic=target,
        semantic=_single_face_flip_semantic(),
    )
    config = MultiviewNasalObjectiveConfig(
        front_image_weight=0.0,
        side_image_weight=25.0,
        flame_prior_weight=0.0,
        semantic_prior_weight=0.0,
        smoothness_weight=0.0,
        symmetry_weight=0.0,
    )
    target_theta = np.r_[np.zeros(context.observable_rank), target]
    target_result = evaluate_multiview_nasal_objective(
        target_theta,
        context,
        config,
    )
    baseline_result = evaluate_multiview_nasal_objective(
        np.zeros(context.parameter_count),
        context,
        config,
    )

    result = fit_multiview_nasal_shape(
        context,
        objective_config=config,
        optimization_config=NasalOptimizationConfig(max_nfev=200),
    )

    assert target_result.report_data["surface_orientation_barrier"][
        "min_signed_area_ratio"
    ] < 0.0
    assert result.success
    assert result.final_objective is not None
    assert abs(result.coefficients[context.observable_rank]) > 0.05
    assert sum(
        result.final_objective.raw_costs[name]
        for name in context.image_term_names
    ) < sum(
        baseline_result.raw_costs[name]
        for name in context.image_term_names
    )
    assert result.final_objective.report_data[
        "surface_orientation_barrier"
    ]["min_signed_area_ratio"] > 0.0
    assert (
        result.final_objective.raw_costs["surface_orientation_barrier"]
        > 0.0
    )


def test_best_so_far_trace_is_monotonic_and_term_costs_sum(monkeypatch):
    context, _ = _objective_problem()
    target = np.linspace(-0.2, 0.2, context.parameter_count)
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        _synthetic_evaluator(
            context,
            lambda coefficients: coefficients - target,
        ),
    )

    result = fit_multiview_nasal_shape(
        context,
        initial_coefficients=np.full(context.parameter_count, 0.1),
    )

    costs = np.asarray(
        [item.total_robust_cost for item in result.iteration_trace]
    )
    assert len(costs) >= 2
    assert np.all(np.diff(costs) <= 1e-12)
    for index, item in enumerate(result.iteration_trace):
        assert sum(item.robust_term_costs.values()) == pytest.approx(
            item.total_robust_cost,
            abs=1e-12,
        )
        assert sum(
            values[index]
            for values in result.objective_term_trace.values()
        ) == pytest.approx(item.total_robust_cost, abs=1e-12)


@pytest.mark.parametrize(
    "initial",
    [
        lambda count: np.r_[np.nan, np.zeros(count - 1)],
        lambda count: np.zeros(count - 1),
        lambda count: np.full(count, 3.01),
    ],
)
def test_invalid_initial_coefficients_return_clear_failure(initial):
    context, _ = _objective_problem()

    result = fit_multiview_nasal_shape(
        context,
        initial_coefficients=initial(context.parameter_count),
    )

    assert not result.success
    assert result.failure_reason == "invalid_initial_coefficients"
    assert result.baseline_unchanged is True
    assert result.nfev == 0
    assert result.objective_evaluation_count == 0
    assert result.final_objective is None


@pytest.mark.parametrize("behavior", ["nan", "exception"])
def test_invalid_objective_evaluation_returns_failure(monkeypatch, behavior):
    context, _ = _objective_problem()
    template = _synthetic_evaluator(
        context,
        lambda coefficients: coefficients,
    )

    def broken(*args):
        if behavior == "exception":
            raise RuntimeError("synthetic evaluation failure")
        evaluated = template(*args)
        values = np.array(evaluated.residuals, copy=True)
        values[0] = np.nan
        return SimpleNamespace(
            residuals=values,
            term_residuals=evaluated.term_residuals,
        )

    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        broken,
    )

    result = fit_multiview_nasal_shape(context)

    assert not result.success
    assert result.failure_reason == "objective_evaluation_failed"
    assert result.baseline_unchanged is True
    assert "objective evaluation failed" in result.solver_message
    assert result.objective_evaluation_count == 1


def test_no_positive_confidence_evidence_returns_failure():
    context, _ = _objective_problem(
        confidence={
            "front": 0.0,
            "subject-left": 0.0,
            "subject-right": 0.0,
        }
    )

    result = fit_multiview_nasal_shape(context)

    assert not result.success
    assert result.failure_reason == "no_effective_observations"
    assert result.baseline_unchanged is True
    assert result.report["effective_observation_counts"] == {
        name: 0 for name in context.image_term_names
    }
    assert result.objective_evaluation_count == 0


def test_zero_image_weights_disable_all_effective_evidence(monkeypatch):
    context, _ = _objective_problem()

    def forbidden_solver(*_args, **_kwargs):
        raise AssertionError("solver must not run without enabled evidence")

    monkeypatch.setattr(nasal_optimizer, "least_squares", forbidden_solver)
    result = fit_multiview_nasal_shape(
        context,
        objective_config=MultiviewNasalObjectiveConfig(
            front_image_weight=0.0,
            side_image_weight=0.0,
        ),
    )

    assert not result.success
    assert result.failure_reason == "no_effective_observations"
    assert all(
        value > 0
        for value in result.report[
            "raw_effective_observation_counts"
        ].values()
    )
    assert all(
        value == 0.0
        for value in result.report[
            "weighted_effective_observation_counts"
        ].values()
    )
    assert all(
        value == 0.0
        for value in result.report[
            "weighted_effective_confidence_sums"
        ].values()
    )


def test_solver_success_is_not_overridden_by_nonzero_low_rank_problem(
    monkeypatch,
):
    context, _ = _objective_problem()
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        _synthetic_evaluator(
            context,
            lambda _coefficients: np.ones(context.parameter_count),
        ),
    )

    result = fit_multiview_nasal_shape(context)

    assert result.success
    assert result.failure_reason is None
    assert result.jacobian_rank == 0
    assert result.final_objective is not None
    assert result.baseline_unchanged is True


def test_unsuccessful_rank_deficient_solver_is_classified_singular(
    monkeypatch,
):
    context, _ = _objective_problem()

    def unsuccessful(fun, x0, **_kwargs):
        fun(x0)
        return _solver_result(
            context,
            x=np.asarray(x0),
            success=False,
            status=0,
            message="maximum evaluations exceeded",
            jac=np.zeros(
                (context.parameter_count, context.parameter_count),
                dtype=np.float64,
            ),
        )

    monkeypatch.setattr(nasal_optimizer, "least_squares", unsuccessful)

    result = fit_multiview_nasal_shape(context)

    assert not result.success
    assert result.failure_reason == "singular_jacobian"
    assert result.jacobian_rank == 0
    assert result.report["selected_coefficients_source"] == "solver_final"


def test_solver_exception_returns_failure_with_available_trace(monkeypatch):
    context, _ = _objective_problem()

    def broken_solver(fun, x0, **_kwargs):
        fun(x0)
        raise RuntimeError("synthetic solver failure")

    monkeypatch.setattr(nasal_optimizer, "least_squares", broken_solver)

    result = fit_multiview_nasal_shape(context)

    assert not result.success
    assert result.failure_reason == "solver_failed"
    assert result.baseline_unchanged is True
    assert result.objective_evaluation_count == 1
    assert len(result.iteration_trace) == 1
    assert result.final_objective is not None
    assert "synthetic solver failure" in result.solver_message


def test_solver_endpoint_wins_over_lower_cost_finite_difference_probe(
    monkeypatch,
):
    context, _ = _objective_problem()
    probe = np.full(context.parameter_count, 0.2)
    solved_x = np.zeros(context.parameter_count)
    synthetic = _synthetic_evaluator(
        context,
        lambda coefficients: coefficients - probe,
    )
    full_coefficients = []
    real_full = evaluate_multiview_nasal_objective

    def solver(fun, _x0, **_kwargs):
        fun(probe)
        return _solver_result(
            context,
            x=solved_x,
            nfev=1,
            njev=1,
        )

    def captured_full(coefficients, *args):
        full_coefficients.append(np.array(coefficients, copy=True))
        return real_full(coefficients, *args)

    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        synthetic,
    )
    monkeypatch.setattr(nasal_optimizer, "least_squares", solver)
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective",
        captured_full,
    )

    result = fit_multiview_nasal_shape(context)

    assert result.success
    np.testing.assert_array_equal(result.coefficients, solved_x)
    np.testing.assert_array_equal(full_coefficients, [solved_x])
    np.testing.assert_array_equal(
        result.final_objective.candidate.vertices,
        context.baseline_vertices,
    )
    np.testing.assert_array_equal(
        result.iteration_trace[0].coefficients,
        probe,
    )
    assert result.report["selected_coefficients_source"] == "solver_final"
    assert result.report["final_objective_role"] == "solver_endpoint"
    assert result.objective_evaluation_count == 1
    assert result.nfev == 1


def test_active_bound_low_rank_scipy_success_remains_success(monkeypatch):
    context, _ = _objective_problem()
    endpoint = np.zeros(context.parameter_count)
    endpoint[0] = context.parameter_lower_bounds[0]
    active_mask = np.zeros(context.parameter_count, dtype=np.int64)
    active_mask[0] = -1
    jacobian = np.eye(context.parameter_count, dtype=np.float64)
    jacobian[-1] = 0.0

    def solver(fun, x0, **_kwargs):
        fun(x0)
        return _solver_result(
            context,
            x=endpoint,
            active_mask=active_mask,
            jac=jacobian,
        )

    monkeypatch.setattr(nasal_optimizer, "least_squares", solver)
    result = fit_multiview_nasal_shape(context)

    assert result.success
    np.testing.assert_array_equal(result.coefficients, endpoint)
    np.testing.assert_array_equal(result.active_mask, active_mask)
    assert result.jacobian_rank == context.parameter_count - 1
    assert result.failure_reason is None


@pytest.mark.parametrize(
    ("field", "bad_value", "error_fragment"),
    [
        (
            "x",
            lambda context: np.full(context.parameter_count, np.nan),
            "x:",
        ),
        ("optimality", lambda _context: np.nan, "optimality:"),
        (
            "active_mask",
            lambda context: np.full(context.parameter_count, 7),
            "active_mask:",
        ),
        ("nfev", lambda _context: "many", "nfev:"),
        ("njev", lambda _context: -1, "njev:"),
    ],
)
def test_malformed_solver_result_returns_failure_without_raising(
    monkeypatch,
    field,
    bad_value,
    error_fragment,
):
    context, _ = _objective_problem()

    def malformed(fun, x0, **_kwargs):
        fun(x0)
        return _solver_result(
            context,
            **{field: bad_value(context)},
        )

    monkeypatch.setattr(nasal_optimizer, "least_squares", malformed)

    result = fit_multiview_nasal_shape(context)

    assert not result.success
    assert result.failure_reason == "solver_failed"
    assert result.baseline_unchanged is True
    assert any(
        error_fragment in error
        for error in result.report["diagnostic_parse_errors"]
    )
    assert "invalid solver result" in result.solver_message
    if field == "x":
        np.testing.assert_array_equal(
            result.coefficients,
            np.zeros(context.parameter_count),
        )
        assert (
            result.report["selected_coefficients_source"]
            == "initial_diagnostic_fallback"
        )
    else:
        assert result.report["selected_coefficients_source"] == "solver_final"


def test_solver_endpoint_full_evaluation_failure_is_captured(monkeypatch):
    context, _ = _objective_problem()
    endpoint = np.full(context.parameter_count, 0.1)

    def solver(fun, x0, **_kwargs):
        fun(x0)
        return _solver_result(context, x=endpoint)

    def broken_full(*_args, **_kwargs):
        raise RuntimeError("synthetic full evaluation failure")

    monkeypatch.setattr(nasal_optimizer, "least_squares", solver)
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective",
        broken_full,
    )

    result = fit_multiview_nasal_shape(context)

    assert not result.success
    assert result.failure_reason == "objective_evaluation_failed"
    np.testing.assert_array_equal(result.coefficients, endpoint)
    assert result.final_objective is None
    assert result.report["selected_coefficients_source"] == "solver_final"
    assert "synthetic full evaluation failure" in result.solver_message


def test_scipy_receives_fixed_numerical_contract(monkeypatch):
    flame_scales = np.array([0.4, 1.7])
    context, _ = _objective_problem(
        flame_standard_deviations=flame_scales,
    )
    semantic_scales = tuple(np.linspace(0.5, 1.2, 8))
    objective_config = MultiviewNasalObjectiveConfig(
        semantic_prior_standard_deviations=semantic_scales,
        robust_f_scale=2.5,
    )
    optimization_config = NasalOptimizationConfig(
        max_nfev=17,
        diff_step=1e-5,
    )
    captured = {}

    def spy(fun, x0, **kwargs):
        captured["x0"] = x0
        captured.update(kwargs)
        residuals = fun(x0)
        count = len(x0)
        return _solver_result(
            context,
            x=np.asarray(x0),
            message="spy converged",
            jac=np.eye(count, dtype=np.float64),
        )

    monkeypatch.setattr(nasal_optimizer, "least_squares", spy)

    result = fit_multiview_nasal_shape(
        context,
        objective_config=objective_config,
        optimization_config=optimization_config,
    )

    assert result.success
    assert captured["jac"] == "2-point"
    assert captured["method"] == "trf"
    assert captured["bounds"][0] is context.parameter_lower_bounds
    assert captured["bounds"][1] is context.parameter_upper_bounds
    np.testing.assert_array_equal(
        captured["bounds"][0],
        np.full(context.parameter_count, -3.0),
    )
    np.testing.assert_array_equal(
        captured["bounds"][1],
        np.full(context.parameter_count, 3.0),
    )
    assert captured["loss"] == objective_config.robust_loss
    assert captured["f_scale"] == objective_config.robust_f_scale
    np.testing.assert_allclose(
        captured["x_scale"],
        np.r_[flame_scales, semantic_scales],
    )
    assert captured["diff_step"] == optimization_config.diff_step
    assert captured["max_nfev"] == optimization_config.max_nfev


def test_iterative_path_uses_residual_only_and_full_evaluator_once(
    monkeypatch,
):
    context, _ = _objective_problem()
    target = np.linspace(-0.25, 0.25, context.parameter_count)
    synthetic = _synthetic_evaluator(
        context,
        lambda coefficients: coefficients - target,
    )
    real_full = evaluate_multiview_nasal_objective
    residual_calls = 0
    full_calls = 0

    def counted_residual(*args):
        nonlocal residual_calls
        residual_calls += 1
        return synthetic(*args)

    def counted_full(*args):
        nonlocal full_calls
        full_calls += 1
        return real_full(*args)

    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        counted_residual,
    )
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective",
        counted_full,
    )

    result = fit_multiview_nasal_shape(context)

    assert result.success
    assert residual_calls == result.objective_evaluation_count
    assert residual_calls >= context.parameter_count + 2
    assert full_calls == 1


def test_result_trace_and_nested_report_are_deeply_immutable(monkeypatch):
    context, _ = _objective_problem()
    target = np.linspace(-0.1, 0.1, context.parameter_count)
    monkeypatch.setattr(
        nasal_optimizer,
        "evaluate_multiview_nasal_objective_residuals",
        _synthetic_evaluator(
            context,
            lambda coefficients: coefficients - target,
        ),
    )
    result = fit_multiview_nasal_shape(context)

    with pytest.raises(ValueError):
        result.coefficients[0] = 9.0
    with pytest.raises(ValueError):
        result.active_mask[0] = 1
    with pytest.raises(ValueError):
        result.iteration_trace[0].coefficients[0] = 9.0
    with pytest.raises(TypeError):
        result.iteration_trace[0].robust_term_costs["new"] = 1.0
    with pytest.raises(TypeError):
        result.objective_term_trace["new"] = ()
    with pytest.raises(TypeError):
        result.report["bounds"]["lower"] = ()
    with pytest.raises(FrozenInstanceError):
        result.success = False


@pytest.mark.parametrize(
    "kwargs",
    [
        {"max_nfev": 0},
        {"ftol": 0.0},
        {"diff_step": np.nan},
        {"trace_cost_tolerance": -1.0},
        {"method": "dogbox"},
        {"jac": "3-point"},
    ],
)
def test_optimization_config_is_strict(kwargs):
    with pytest.raises(ValueError):
        NasalOptimizationConfig(**kwargs)


def test_trace_dataclass_rejects_inconsistent_term_total():
    with pytest.raises(ValueError):
        NasalOptimizationIteration(
            function_evaluation=1,
            coefficients=np.zeros(2),
            total_robust_cost=2.0,
            robust_term_costs={"term": 1.0},
        )

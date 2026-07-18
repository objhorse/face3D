import numpy as np
import pytest

from src.geometry.identity_quality import (
    IdentityDriftThresholds,
    make_identity_drift_gate,
    mica_centered_shape_regularization,
    select_identity_safe_candidate,
    select_stable_refinement_checkpoint,
)


def _mesh(scale=1.0):
    return np.array(
        [
            [-1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ],
        dtype=np.float32,
    ) * scale


def test_shape_regularizer_prefers_mica_identity_over_zero_shape():
    anchor = np.array([2.0, -1.0, 0.5], dtype=np.float32)
    at_anchor = mica_centered_shape_regularization(
        anchor,
        anchor,
        anchor_weight=5e-4,
        mean_shape_weight=5e-6,
    )
    at_zero = mica_centered_shape_regularization(
        np.zeros_like(anchor),
        anchor,
        anchor_weight=5e-4,
        mean_shape_weight=5e-6,
    )

    assert at_anchor < at_zero


def test_identity_gate_accepts_small_mica_relative_drift():
    anchor_shape = np.zeros(4, dtype=np.float32)
    candidate_shape = np.array([0.5, 0.5, 0.0, 0.0], dtype=np.float32)
    anchor_mesh = _mesh()
    candidate_mesh = anchor_mesh + np.array([0.01, 0.0, 0.0], dtype=np.float32)

    gate = make_identity_drift_gate(
        anchor_shape=anchor_shape,
        candidate_shape=candidate_shape,
        anchor_vertices=anchor_mesh,
        candidate_vertices=candidate_mesh,
    )

    assert gate["passed"]
    assert gate["metrics"]["mean_displacement_pct"] == pytest.approx(0.5)


def test_identity_gate_rejects_parameter_drift_even_when_mesh_is_small():
    gate = make_identity_drift_gate(
        anchor_shape=np.zeros(4, dtype=np.float32),
        candidate_shape=np.array([8.0, 0.0, 0.0, 0.0], dtype=np.float32),
        anchor_vertices=_mesh(),
        candidate_vertices=_mesh(),
    )

    assert not gate["passed"]
    assert "coefficient_delta_l2_exceeded" in gate["issues"]


def test_identity_gate_rejects_geometry_drift_even_when_coefficients_pass():
    anchor_mesh = _mesh()
    candidate_mesh = anchor_mesh.copy()
    candidate_mesh[0, 2] += 0.12
    gate = make_identity_drift_gate(
        anchor_shape=np.zeros(4, dtype=np.float32),
        candidate_shape=np.ones(4, dtype=np.float32),
        anchor_vertices=anchor_mesh,
        candidate_vertices=candidate_mesh,
        thresholds=IdentityDriftThresholds(max_displacement_pct=4.0),
    )

    assert not gate["passed"]
    assert "max_neutral_mesh_displacement_exceeded" in gate["issues"]


def test_candidate_selection_excludes_failed_identity_and_mesh_gates():
    candidates = [
        {
            "attempt": 1,
            "observation_score_px": 1.0,
            "identity_gate": {"passed": False, "metrics": {}},
            "mesh_quality_gate": {"passed": True},
        },
        {
            "attempt": 2,
            "observation_score_px": 2.0,
            "identity_gate": {
                "passed": True,
                "metrics": {"mean_displacement_pct": 1.0, "coefficient_delta_l2": 5.0},
            },
            "mesh_quality_gate": {"passed": True},
        },
    ]

    assert select_identity_safe_candidate(candidates)["attempt"] == 2


def test_candidate_selection_prefers_lower_drift_within_pixel_tolerance():
    candidates = [
        {
            "attempt": 1,
            "observation_score_px": 5.0,
            "identity_gate": {
                "passed": True,
                "metrics": {"mean_displacement_pct": 1.2, "coefficient_delta_l2": 5.5},
            },
            "mesh_quality_gate": {"passed": True},
        },
        {
            "attempt": 2,
            "observation_score_px": 5.8,
            "identity_gate": {
                "passed": True,
                "metrics": {"mean_displacement_pct": 0.8, "coefficient_delta_l2": 4.0},
            },
            "mesh_quality_gate": {"passed": True},
        },
    ]

    assert select_identity_safe_candidate(candidates, observation_tolerance=1.0)["attempt"] == 2


def test_refinement_checkpoint_uses_earliest_safe_step_within_render_tolerance():
    candidates = [
        {
            "attempt": 10,
            "step": 10,
            "accepted": True,
            "observation_score_px": 7.05,
            "identity_gate": {
                "passed": True,
                "metrics": {"mean_displacement_pct": 0.94, "coefficient_delta_l2": 4.1},
            },
            "mesh_quality_gate": {"passed": True},
        },
        {
            "attempt": 70,
            "step": 70,
            "accepted": True,
            "observation_score_px": 6.94,
            "identity_gate": {
                "passed": True,
                "metrics": {"mean_displacement_pct": 0.90, "coefficient_delta_l2": 5.2},
            },
            "mesh_quality_gate": {"passed": True},
        },
    ]

    selected = select_stable_refinement_checkpoint(
        candidates, observation_tolerance=0.75
    )
    assert selected["step"] == 10

import numpy as np
import torch

from src.geometry.controlled_identity_deformation import (
    LowFrequencyBasisConfig,
    LowFrequencyOptimizationConfig,
    LowFrequencySafetyThresholds,
    apply_low_frequency_identity_numpy,
    apply_low_frequency_identity_torch,
    build_low_frequency_identity_basis,
    evaluate_low_frequency_observations,
    evaluate_low_frequency_safety,
    optimize_low_frequency_identity,
    restrict_low_frequency_identity_basis,
    select_low_frequency_checkpoint,
)


def _synthetic_face():
    xs = np.linspace(-1.0, 1.0, 21, dtype=np.float32)
    ys = np.linspace(-1.1, 1.1, 23, dtype=np.float32)
    vertices = []
    for y in ys:
        for x in xs:
            z = 0.35 - 0.16 * x * x - 0.05 * y * y
            vertices.append((x, y, z))
    vertices = np.asarray(vertices, dtype=np.float32)
    faces = []
    width = len(xs)
    for row in range(len(ys) - 1):
        for col in range(len(xs) - 1):
            a = row * width + col
            b = a + 1
            c = a + width
            d = c + 1
            faces.extend(((a, b, d), (a, d, c)))
    faces = np.asarray(faces, dtype=np.int64)
    angles = np.linspace(0.0, 2.0 * np.pi, 32, endpoint=False)
    landmarks = np.stack(
        [
            0.92 * np.cos(angles),
            1.02 * np.sin(angles),
            0.20 + 0.03 * np.cos(angles) ** 2,
        ],
        axis=1,
    ).astype(np.float32)
    landmarks = np.concatenate(
        [landmarks, np.array([[0.0, 0.0, 0.40], [-0.3, 0.2, 0.33], [0.3, 0.2, 0.33]])],
        axis=0,
    )
    protected = np.array(
        [
            [-0.28, 0.22, 0.33],
            [0.28, 0.22, 0.33],
            [0.0, 0.02, 0.40],
            [0.0, -0.27, 0.36],
        ],
        dtype=np.float32,
    )
    return vertices, faces, landmarks, protected


def _basis():
    vertices, faces, landmarks, protected = _synthetic_face()
    basis = build_low_frequency_identity_basis(
        vertices,
        faces,
        landmarks,
        protected_points=protected,
        config=LowFrequencyBasisConfig(
            protection_core_ratio=0.055,
            protection_outer_ratio=0.14,
        ),
    )
    return vertices, faces, basis


def test_low_frequency_basis_is_deterministic_and_low_dimensional():
    vertices, faces, landmarks, protected = _synthetic_face()
    first = build_low_frequency_identity_basis(vertices, faces, landmarks, protected)
    second = build_low_frequency_identity_basis(vertices, faces, landmarks, protected)

    assert len(first.names) == 16
    assert first.names == second.names
    np.testing.assert_array_equal(first.vectors, second.vectors)
    assert first.vectors.shape == (16, len(vertices), 3)
    assert np.isfinite(first.vectors).all()
    assert np.any(first.editable_mask)


def test_paired_width_controls_are_mirrored():
    vertices, _faces, basis = _basis()
    row_width = 21
    y_row = 15
    xneg_vertex = y_row * row_width + 5
    xpos_vertex = y_row * row_width + 15
    left = basis.names.index("cheek_width_xneg")
    right = basis.names.index("cheek_width_xpos")

    np.testing.assert_allclose(
        basis.vectors[left, xneg_vertex, 0],
        -basis.vectors[right, xpos_vertex, 0],
        rtol=1e-5,
        atol=1e-7,
    )
    np.testing.assert_allclose(basis.vectors[left, xneg_vertex, 1:], 0.0, atol=1e-8)
    np.testing.assert_allclose(basis.vectors[right, xpos_vertex, 1:], 0.0, atol=1e-8)


def test_protected_feature_cores_have_zero_basis_support():
    _vertices, _faces, basis = _basis()

    assert np.any(basis.protected_mask)
    assert np.all(basis.vectors[:, basis.protected_mask, :] == 0.0)
    editable_unprotected = basis.editable_mask & ~basis.protected_mask
    assert np.any(np.linalg.norm(basis.vectors[:, editable_unprotected, :], axis=2) > 0.0)


def test_basis_restriction_locks_lower_face_controls():
    _vertices, _faces, basis = _basis()

    restricted, active = restrict_low_frequency_identity_basis(
        basis, ("temple_", "cheekbone_", "cheek_")
    )

    assert active
    for index, name in enumerate(restricted.names):
        if name.startswith(("jaw_", "chin_")):
            assert np.all(restricted.vectors[index] == 0.0)


def test_numpy_application_preserves_topology_and_clamps_displacement():
    vertices, faces, basis = _basis()
    original_faces = faces.copy()
    coefficients = np.ones(len(basis.names), dtype=np.float32)
    candidate, displacement = apply_low_frequency_identity_numpy(vertices, basis, coefficients)
    clipped_candidate, _ = apply_low_frequency_identity_numpy(vertices, basis, coefficients * 4.0)

    assert candidate.shape == vertices.shape
    np.testing.assert_array_equal(faces, original_faces)
    assert np.linalg.norm(displacement, axis=1).max() <= basis.max_vertex_displacement + 1e-6
    np.testing.assert_allclose(displacement[basis.protected_mask], 0.0, atol=1e-8)
    np.testing.assert_allclose(candidate, clipped_candidate, rtol=1e-6, atol=1e-7)


def test_torch_application_matches_numpy_and_has_finite_gradient():
    vertices, _faces, basis = _basis()
    coefficients = np.linspace(-0.6, 0.6, len(basis.names), dtype=np.float32)
    expected, _ = apply_low_frequency_identity_numpy(vertices, basis, coefficients)

    baseline_t = torch.tensor(vertices)
    vectors_t = torch.tensor(basis.vectors)
    coeff_t = torch.tensor(coefficients, requires_grad=True)
    protected_t = torch.tensor(basis.protected_mask)
    candidate_t, displacement_t = apply_low_frequency_identity_torch(
        baseline_t,
        vectors_t,
        coeff_t,
        basis.max_vertex_displacement,
        protected_t,
    )
    loss = candidate_t[:, 0].square().mean() + displacement_t[:, 2].square().mean()
    loss.backward()

    np.testing.assert_allclose(candidate_t.detach().numpy(), expected, rtol=1e-5, atol=1e-6)
    assert coeff_t.grad is not None
    assert torch.isfinite(coeff_t.grad).all()


def test_optimizer_improves_symmetric_width_target_with_bounded_controls():
    vertices, faces, basis = _basis()
    baseline_t = torch.tensor(vertices)
    target_x = baseline_t[:, 0] * 1.025
    editable = torch.tensor(basis.editable_mask & ~basis.protected_mask)

    def observation_loss(candidate):
        error = (candidate[editable, 0] - target_x[editable]) / basis.face_width
        return error.square().mean()

    before = float(observation_loss(baseline_t))
    result = optimize_low_frequency_identity(
        vertices,
        faces,
        basis,
        observation_loss,
        config=LowFrequencyOptimizationConfig(
            max_iterations=35,
            learning_rate=0.08,
            checkpoint_interval=5,
            observation_weight=100.0,
        ),
        device="cpu",
    )
    final = result["checkpoints"][-1]
    after = float(observation_loss(torch.tensor(final["vertices"])))

    assert result["parameter_count"] == 16
    assert after < before * 0.5
    assert np.max(np.abs(final["coefficients"])) <= 1.0
    assert final["losses"]["symmetry_loss"] < 1e-3


def test_safety_gate_rejects_protected_motion_and_large_mean_motion():
    vertices, faces, basis = _basis()
    clean = evaluate_low_frequency_safety(vertices, vertices.copy(), faces, basis)
    assert clean["passed"]

    moved = vertices.copy()
    moved[basis.protected_mask, 2] += basis.face_width * 0.01
    decision = evaluate_low_frequency_safety(
        vertices,
        moved,
        faces,
        basis,
        thresholds=LowFrequencySafetyThresholds(
            max_vertex_displacement_ratio=0.10,
            max_mean_displacement_ratio=0.10,
            max_protected_displacement_ratio=0.001,
        ),
    )
    assert not decision["passed"]
    assert not decision["gates"]["protected_displacement"]


def test_checkpoint_selection_returns_exact_baseline_when_all_rejected():
    vertices, _faces, basis = _basis()
    checkpoints = []
    for step, score in ((5, 3.0), (10, 2.0)):
        candidate, displacement = apply_low_frequency_identity_numpy(
            vertices,
            basis,
            np.full(len(basis.names), 0.1 * step / 5.0, dtype=np.float32),
        )
        checkpoints.append(
            {
                "step": step,
                "vertices": candidate,
                "displacement": displacement,
                "coefficients": np.zeros(len(basis.names), dtype=np.float32),
                "losses": {"coefficient_norm": 0.0},
                "mock_score": score,
            }
        )

    selected, selected_trial, trials = select_low_frequency_checkpoint(
        vertices,
        checkpoints,
        lambda checkpoint: {
            "accepted": False,
            "rank_score": checkpoint["mock_score"],
            "reason": "synthetic rejection",
        },
    )

    np.testing.assert_array_equal(selected, vertices)
    assert selected_trial is None
    assert len(trials) == 2


def test_observation_gates_require_front_gain_and_preserve_each_profile():
    def record(view, boundary, interior=10.0, dice=0.90):
        return {
            "view": view,
            "interior_mean_px": interior,
            "silhouette": {
                "trusted_boundary_mean_px": boundary,
                "trusted_region_dice": dice,
            },
        }

    before = [record("front", 30.0), record("left", 20.0), record("right", 20.0)]
    accepted_after = [
        record("front", 20.0, 10.5),
        record("left", 21.0, 10.5),
        record("right", 19.5, 10.5),
    ]
    accepted = evaluate_low_frequency_observations(
        before,
        accepted_after,
        safety_gate={"passed": True},
        mesh_quality_gate={"passed": True},
    )
    assert accepted["accepted"]

    broken_profile = list(accepted_after)
    broken_profile[1] = record("left", 23.0, 10.5)
    rejected = evaluate_low_frequency_observations(
        before,
        broken_profile,
        safety_gate={"passed": True},
        mesh_quality_gate={"passed": True},
    )
    assert not rejected["accepted"]
    assert "profiles_preserved" in rejected["failed_gates"]


def test_observation_gate_rejects_aggregate_gain_that_damages_one_front_region():
    def record(view, boundary, regions=None):
        return {
            "view": view,
            "interior_mean_px": 10.0,
            "silhouette": {
                "trusted_boundary_mean_px": boundary,
                "regional_balanced_boundary_mean_px": boundary,
                "trusted_region_dice": 0.90,
                "regions": regions or {},
            },
        }

    front_before = {
        "forehead": {"boundary_mean_px": 20.0, "signed_width_error_px": -10.0},
        "cheek": {"boundary_mean_px": 20.0, "signed_width_error_px": -20.0},
        "temple": {"boundary_mean_px": 20.0, "signed_width_error_px": -2.0},
    }
    front_after = {
        "forehead": {"boundary_mean_px": 10.0, "signed_width_error_px": -4.0},
        "cheek": {"boundary_mean_px": 10.0, "signed_width_error_px": -8.0},
        "temple": {"boundary_mean_px": 24.0, "signed_width_error_px": 12.0},
    }
    before = [
        record("front", 20.0, front_before),
        record("left", 20.0),
        record("right", 20.0),
    ]
    after = [
        record("front", 13.0, front_after),
        record("left", 20.0),
        record("right", 20.0),
    ]

    decision = evaluate_low_frequency_observations(
        before,
        after,
        safety_gate={"passed": True},
        mesh_quality_gate={"passed": True},
    )

    assert not decision["accepted"]
    assert "front_regions_preserved" in decision["failed_gates"]
    assert decision["metrics"]["max_front_region_worsen_ratio"] == 0.2

from __future__ import annotations

import numpy as np

from src.geometry.eyelid_fit import project_landmarks
from src.geometry.semantic_eyelid_optimizer import (
    EYELID_PARAMETER_NAMES,
    SemanticEyelidOptimizationConfig,
    apply_semantic_eyelid_parameters,
    control_parameters_from_semantics,
    fit_semantic_eyelids_stage_c,
)
from src.geometry.semantic_eyelid_rig import build_semantic_eyelid_rig


def _two_eye_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for eye_offset in (0.0, 10.0):
        start = len(vertices)
        for y in range(5):
            for x in range(5):
                vertices.append(
                    [eye_offset + x, y, 1.0 + 0.05 * (2 - abs(x - 2))]
                )
        for y in range(4):
            for x in range(4):
                a = start + y * 5 + x
                faces.extend(([a, a + 1, a + 5], [a + 1, a + 6, a + 5]))
    vertices.append([100.0, 100.0, 100.0])
    mapping = np.zeros((68, 3), dtype=np.int64)
    right = {
        36: [5, 10, 6],
        37: [10, 11, 6],
        38: [11, 12, 7],
        39: [8, 13, 12],
        40: [12, 17, 13],
        41: [11, 16, 12],
    }
    left = {
        index + 6: [vertex + 25 for vertex in triangle]
        for index, triangle in right.items()
    }
    for index, triangle in {**right, **left}.items():
        mapping[index] = triangle
    return (
        np.asarray(vertices, dtype=np.float64),
        np.asarray(faces, dtype=np.int64),
        mapping,
    )


def _camera() -> dict:
    return {
        "K": np.array(
            [[120.0, 0.0, 80.0], [0.0, 120.0, 60.0], [0.0, 0.0, 1.0]]
        ),
        "R": np.eye(3),
        "t": np.array([0.0, 0.0, 12.0]),
    }


def test_semantic_parameterization_keeps_corners_fixed() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=3,
        freeze_corner_seeds=True,
    )
    coefficients = np.array(
        [0.05, -0.10, 0.02, -0.03, -0.08, 0.01],
        dtype=np.float64,
    )
    candidate, controls = apply_semantic_eyelid_parameters(
        vertices,
        rig,
        coefficients,
    )

    assert controls.shape == (8, 2)
    for index, name in enumerate(rig.control_names):
        if name.endswith("_corner"):
            np.testing.assert_array_equal(controls[index], 0.0)
    np.testing.assert_array_equal(
        candidate[rig.protected_vertices],
        vertices[rig.protected_vertices],
    )


def test_stage_c_recovers_synthetic_local_eye_shape() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=3,
        freeze_corner_seeds=True,
    )
    barycentric = np.full((68, 3), 1.0 / 3.0, dtype=np.float64)
    face_width = float(np.ptp(vertices[:, 0]))
    target_coefficients = np.array(
        [
            0.0005 * face_width,
            -0.0015 * face_width,
            0.0003 * face_width,
            -0.0004 * face_width,
            -0.0012 * face_width,
            0.0002 * face_width,
        ],
        dtype=np.float64,
    )
    target_vertices, _controls = apply_semantic_eyelid_parameters(
        vertices,
        rig,
        target_coefficients,
    )
    camera = _camera()
    target = project_landmarks(target_vertices, mapping, barycentric, camera)
    cameras = {name: camera for name in ("front", "subject-left", "subject-right")}
    observed = {name: target.copy() for name in cameras}
    weights = {
        name: {"subject_right": 1.0, "subject_left": 1.0}
        for name in cameras
    }

    result = fit_semantic_eyelids_stage_c(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=mapping,
        barycentric=barycentric,
        cameras=cameras,
        observed_landmarks=observed,
        eye_states={"subject_right": "closed", "subject_left": "closed"},
        view_eye_weights=weights,
        cfg=SemanticEyelidOptimizationConfig(
            parameter_prior_weight=0.001,
            symmetry_weight=0.0,
            deformation_smoothness_weight=0.001,
            max_nfev=200,
        ),
    )

    assert result.report["after"]["mean_px"] < result.report["before"]["mean_px"]
    outside = np.setdiff1d(np.arange(len(vertices)), rig.active_vertices)
    np.testing.assert_array_equal(result.vertices[outside], vertices[outside])
    np.testing.assert_array_equal(
        result.vertices[rig.protected_vertices],
        vertices[rig.protected_vertices],
    )


def test_global_image_translation_does_not_change_semantic_fit() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(
        vertices,
        faces,
        mapping,
        support_rings=3,
        freeze_corner_seeds=True,
    )
    barycentric = np.full((68, 3), 1.0 / 3.0, dtype=np.float64)
    projected = project_landmarks(vertices, mapping, barycentric, _camera())
    observed = projected + np.array([17.0, -11.0], dtype=np.float64)
    weights = {"front": {"subject_right": 1.0, "subject_left": 1.0}}

    result = fit_semantic_eyelids_stage_c(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=mapping,
        barycentric=barycentric,
        cameras={"front": _camera()},
        observed_landmarks={"front": observed},
        eye_states={"subject_right": "open", "subject_left": "open"},
        view_eye_weights=weights,
        cfg=SemanticEyelidOptimizationConfig(max_nfev=80),
    )

    np.testing.assert_allclose(result.coefficients, 0.0, atol=1e-8)
    np.testing.assert_allclose(result.vertices, vertices, atol=1e-8)


def test_parameter_contract_has_exactly_six_semantic_values() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    values = np.zeros(len(EYELID_PARAMETER_NAMES), dtype=np.float64)

    controls = control_parameters_from_semantics(rig, values)

    assert len(EYELID_PARAMETER_NAMES) == 6
    assert controls.shape == (8, 2)

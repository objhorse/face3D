from __future__ import annotations

import numpy as np

from src.geometry.eyelid_fit import (
    EyelidFitConfig,
    control_offsets_from_parameters,
    fit_semantic_eyelids,
    project_landmarks,
)
from src.geometry.semantic_eyelid_rig import build_semantic_eyelid_rig


def _two_eye_mesh() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    vertices = []
    faces = []
    for eye_offset in (0.0, 10.0):
        start = len(vertices)
        for y in range(3):
            for x in range(3):
                vertices.append([eye_offset + x, y, 0.1 * (1 - abs(x - 1))])
        for y in range(2):
            for x in range(2):
                a = start + y * 3 + x
                faces.extend(([a, a + 1, a + 3], [a + 1, a + 4, a + 3]))
    vertices.append([100.0, 100.0, 100.0])
    mapping = np.zeros((68, 3), dtype=np.int64)
    right_triangles = {
        36: [0, 3, 1], 37: [3, 4, 1], 38: [4, 5, 2],
        39: [2, 5, 4], 40: [4, 8, 5], 41: [3, 7, 4],
    }
    left_triangles = {
        42: [9, 12, 10], 43: [12, 13, 10], 44: [13, 14, 11],
        45: [11, 14, 13], 46: [13, 17, 14], 47: [12, 16, 13],
    }
    for index, triangle in {**right_triangles, **left_triangles}.items():
        mapping[index] = triangle
    return (
        np.asarray(vertices, dtype=np.float32),
        np.asarray(faces, dtype=np.int32),
        mapping,
    )


def _camera() -> dict:
    return {
        "K": np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]]),
        "R": np.eye(3),
        "t": np.array([0.0, 0.0, 10.0]),
    }


def test_parameterization_is_low_dimensional_and_compact() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    parameters = np.zeros((8, 2), dtype=np.float32)
    parameters[:, 0] = 0.1
    parameters[:, 1] = 0.05
    offsets = control_offsets_from_parameters(rig, parameters)
    candidate = rig.apply(vertices, offsets)
    outside = np.setdiff1d(np.arange(len(vertices)), rig.active_vertices)

    assert offsets.shape == (8, 3)
    np.testing.assert_array_equal(candidate[outside], vertices[outside])


def test_fit_recovers_synthetic_multiview_eye_target() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    vertices = vertices.copy()
    vertices[:, 2] += 1.0
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    barycentric = np.full((68, 3), 1.0 / 3.0, dtype=np.float32)
    target_parameters = np.zeros((8, 2), dtype=np.float32)
    for index, name in enumerate(rig.control_names):
        if "upper_lid" in name:
            target_parameters[index, 0] = -0.12
        elif "lower_lid" in name:
            target_parameters[index, 0] = 0.08
    target_vertices = rig.apply(
        vertices,
        control_offsets_from_parameters(rig, target_parameters),
    )
    cameras = {name: _camera() for name in ("left", "front", "right")}
    target = project_landmarks(target_vertices, mapping, barycentric, _camera())
    observations = {name: target.copy() for name in cameras}

    result = fit_semantic_eyelids(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=mapping,
        barycentric=barycentric,
        cameras=cameras,
        observed_landmarks=observations,
        eye_states={"subject_right": "closed", "subject_left": "closed"},
        cfg=EyelidFitConfig(
            max_control_offset=0.3,
            bulge_prior=0.0,
            regularization_weight=0.01,
            symmetry_weight=0.01,
            max_nfev=250,
        ),
    )

    assert result.report["eye_reprojection_after_px"] < result.report["eye_reprojection_before_px"]
    outside = np.setdiff1d(np.arange(len(vertices)), rig.active_vertices)
    np.testing.assert_array_equal(result.vertices[outside], vertices[outside])
    assert result.report["mesh_quality"]["accepted"]


def test_uncertain_state_keeps_baseline() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    barycentric = np.full((68, 3), 1.0 / 3.0, dtype=np.float32)
    cameras = {"front": _camera()}
    observed = {"front": project_landmarks(vertices, mapping, barycentric, _camera())}

    result = fit_semantic_eyelids(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=mapping,
        barycentric=barycentric,
        cameras=cameras,
        observed_landmarks=observed,
        eye_states={"subject_right": "uncertain", "subject_left": "closed"},
    )

    np.testing.assert_array_equal(result.vertices, vertices)
    assert not result.report["accepted"]
    assert result.report["reason"] == "uncertain_eye_state"


def test_global_image_translation_does_not_deform_eyelids() -> None:
    vertices, faces, mapping = _two_eye_mesh()
    vertices = vertices.copy()
    vertices[:, 2] += 1.0
    rig = build_semantic_eyelid_rig(vertices, faces, mapping, support_rings=2)
    barycentric = np.full((68, 3), 1.0 / 3.0, dtype=np.float32)
    camera = _camera()
    projected = project_landmarks(vertices, mapping, barycentric, camera)
    observed = projected + np.array([12.0, -8.0], dtype=np.float64)

    result = fit_semantic_eyelids(
        vertices=vertices,
        faces=faces,
        rig=rig,
        landmark_triangles=mapping,
        barycentric=barycentric,
        cameras={"front": camera},
        observed_landmarks={"front": observed},
        eye_states={"subject_right": "open", "subject_left": "open"},
        cfg=EyelidFitConfig(
            max_control_offset=0.3,
            bulge_prior=0.0,
            regularization_weight=0.01,
            symmetry_weight=0.01,
            max_nfev=100,
        ),
    )

    np.testing.assert_allclose(result.vertices, vertices, atol=1e-7)
    assert result.report["eye_shape_error_before_px"] < 1e-7
